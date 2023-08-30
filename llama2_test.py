# -*-coding:utf-8-*-
import argparse
import json, os
import time
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/chinese-alpaca-2-7b-hf', help="If None, perform inference on the base model")
parser.add_argument('--gpus', type=str, default="0", help='gpu id(default: 0)')
parser.add_argument('--pe', type=str, default="None", help="PE type")
parser.add_argument('--alpha', type=str, default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--temp', type=str, default="llama2_zh", help="Template prompt type")
parser.add_argument('--repeat', type=int, default=1, help="repeat times")
args = parser.parse_args()

# GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Log
os.makedirs("./logs", exist_ok=True)
DATETMIE = time.strftime(r"%m-%d-%H", time.localtime())
LOG_PATH = f"./logs/{DATETMIE}_{args.temp}_pe-{args.pe}.log"
logger.add(LOG_PATH, encoding="utf-8")

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from tensor_parallel import tensor_parallel
from utils import cal_time
from utils import LLaMA2_Buddy_Template, LLaMA2_CH_Template

device = torch.device(0)
model_path = args.model_path


# Postional Embedding
pe_type = args.pe
import pe.src_rope_patch
if pe_type == "rerope":
    import pe.rerope_patch
if pe_type == "leaky_rerope":
    import pe.leaky_rerope_patch
elif pe_type == "ntk_rope_mix":
    import pe.ntk_patch
elif pe_type == "llama_scaling":
    from pe.llama_scaling import apply_ntk_scaling_patch
    apply_ntk_scaling_patch(args.alpha)


# prompt
TEMPLATE = None
if args.temp == "llama2_zh":
    TEMPLATE = LLaMA2_CH_Template()
elif args.temp == "llama2":
    TEMPLATE = LLaMA2_Buddy_Template()
else:
    raise ValueError(f"Unknown temp {args.temp}")


# Sample
question = """请仔细阅读材料，然后回答：
- 菲律宾国家电网公司，中国占股多少？
- 领英计划裁员多少人？
- 吉利德收购Pharmasset的价格是多少？
- 丙肝神药Sovaldi在哪一年上市？
- 中亚峰会将在哪里举行？由谁主持？
- 哪个演员由于侮辱人民军队而被立案调查？
- 哪个项目宣称“能过坦克”的水上道路？
- 如果你是默沙东的CEO，你的首要任务是什么？
- 中行贪污事件中，对余振东的判决是什么？"""
contexts = json.load(open('data/contexts.json')) + json.load(open('data/contexts.100.json'))[:10]
context = '\n\n'.join(contexts)
raw_input_text = '%s\n\n%s' % (context, question)
logger.info(f"input context length: {len(raw_input_text)}")


@cal_time
def load_model():
    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
    )

    # Resize model embeddings
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Vocab of the base model: {model_vocab_size}")
    logger.info(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        logger.info("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)
    model = tensor_parallel(model)
    model.eval()
    return tokenizer, model


@cal_time
def generate(tokenizer, model):
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=500
    )
    with torch.no_grad():
        input_text = TEMPLATE.generate_prompt(raw_input_text)
        
        inputs = tokenizer(input_text, return_tensors="pt").to(device)  #add_special_tokens=False ?
        generation_output = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
        )
        
        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        response = TEMPLATE.split_response(output)
        logger.info(f"Response: \n{response}\n")


def main():
    tokenizer, model = load_model()
    for _ in range(args.repeat):
        generate(tokenizer, model)


if __name__ == "__main__":
    main()