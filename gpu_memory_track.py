# -*-coding:utf-8-*-
import time
import pynvml
import subprocess
import argparse
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default="0", help='log')
args = parser.parse_args()

logger.add("./logs/gpus_memory.log", encoding="utf-8")
logger.info(f"Log: {args.log}")

def gpu_mem_track(log_info):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 启动程序
            process_b = subprocess.Popen(["python", "gpu_memory_track.py", "--log", log_info])

            # 调用
            result = func(*args, **kwargs)

            # 停止程序B
            time.sleep(3)
            process_b.terminate()
            return result
        return wrapper
    return decorator

# @run_program_b_decorator(wait_time=10)  # 在装饰器中传入等待时间参数
# def program_a_function():
#     # 这里是程序A的实际逻辑
#     pass

# # 调用经过装饰的程序A函数
# program_a_function()



def get_total_gpu_memory():
    pynvml.nvmlInit()
    while True:

        device_count = pynvml.nvmlDeviceGetCount()
        total_memory = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory += info.total
        
        logger.info(f"Total GPU Memory: {total_memory / 1024**2} MB")
        time.sleep(5)

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    total_memory = get_total_gpu_memory()
    print(f"Total GPU Memory: {total_memory / 1024**2} MB")
