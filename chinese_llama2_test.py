import argparse
import json, os
import torch
from loguru import logger
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda')
model_path = 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
DEFAULT_LONG_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"""
TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

def generate():
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=400
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
        inputs = tokenizer(input_text,return_tensors="pt").to(device)  #add_special_tokens=False ?
        generation_output = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        print("Response: ",output)