# -*-coding:utf-8-*-
import time
from loguru import logger
from dataclasses import dataclass


def cal_time(func):
    def wrapper(*args, **kwargs):
        """wrapper"""
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        logger.info("[TIME] %s time elapsed: %s secs." % (func.__name__, te - ts))
        return result
    return wrapper


@dataclass
class Template(object):
    TEMPLATE: str
    
    def generate_prompt(self, instruction):
        return instruction
    
    def split_response(self, output):
        return output

@dataclass
class LLaMA2_CH_Template(Template):
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
    DEFAULT_LONG_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"""
    TEMPLATE = (
        "[INST] <<SYS>>\n"
        f"{DEFAULT_SYSTEM_PROMPT}\n"
        "<</SYS>>\n\n"
        "{instruction} [/INST]"
    )
    
    def generate_prompt(self, instruction):
        return self.TEMPLATE.format_map({'instruction': instruction})
    
    def split_response(self, output):
        return output.split("[/INST]")[-1].strip()

@dataclass
class LLaMA2_Buddy_Template(Template):
    TEMPLATE = "User: {instruction}\n\nAssistant"
    
    def generate_prompt(self, instruction):
        return self.TEMPLATE.format_map({'instruction': instruction})
    
    def split_response(self, output):
        return output.split("Assistant")[-1].strip()
