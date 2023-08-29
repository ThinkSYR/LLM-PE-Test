import argparse
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import torch
from loguru import logger
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig

pe_type = "leaky_rerope"

alpha = "auto"
scaling_factor = 4.

import pe.src_rope_patch
if pe_type == "leaky_rerope":
    import pe.leaky_rerope_patch
elif pe_type == "llama_scaling":
    from pe.llama_scaling import apply_ntk_scaling_patch
    apply_ntk_scaling_patch(alpha, scaling_factor)
    

device = torch.device(0)
model_path = 'models/chinese-alpaca-2-7b-hf'


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
DEFAULT_LONG_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"""
TEMPLATE = (
    "[INST] <<SYS>>\n"
    f"{DEFAULT_SYSTEM_PROMPT}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

# 示例Context
question = """请仔细阅读材料，然后回答：
- 菲律宾国家电网公司，中国占股多少？
- 领英计划裁员多少人？
- 吉利德收购Pharmasset的价格是多少？
- 丙肝神药Sovaldi在哪一年上市？
- 中亚峰会将在哪里举行？由谁主持？
- 哪个演员由于侮辱人民军队而被立案调查？
- 哪个项目宣称“能过坦克”的水上道路？
- 如果你是默沙东的CEO，你的首要任务是什么？"""
contexts = json.load(open('data/contexts.json')) + json.load(open('data/contexts.100.json'))[:6]
context = '\n\n'.join(contexts)
raw_input_text = '%s\n\n%s' % (context, question)
print("input length: ", len(raw_input_text))

def generate_prompt(instruction):
    return TEMPLATE.format_map({'instruction': instruction})

def generate():
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=500
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )

    # Resize model embeddings
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)
    model = base_model
    model.eval()

    with torch.no_grad():
        input_text = generate_prompt(instruction=raw_input_text, system_prompt=DEFAULT_SYSTEM_PROMPT)
        inputs = tokenizer(input_text, padding='longest', return_tensors="pt").to(device)  #add_special_tokens=False ?
        generation_output = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        response = output.split("[/INST]")[-1].strip()
        print("Response: ",response)

if __name__ == "__main__":
    generate()