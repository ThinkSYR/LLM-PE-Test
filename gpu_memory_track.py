# -*-coding:utf-8-*-
import time
import pynvml
import subprocess
import argparse
from loguru import logger

logger.add("./logs/gpus_memory.log", encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default="0", help='log')
args = parser.parse_args()

def gpu_mem_track(wait_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 启动程序B
            process_b = subprocess.Popen(["python", "program_b.py"])

            # 调用原始函数
            result = func(*args, **kwargs)

            # 在程序A运行期间，等待指定的时间
            time.sleep(wait_time)

            # 停止程序B
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
    device_count = pynvml.nvmlDeviceGetCount()
    total_memory = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory += info.total

    pynvml.nvmlShutdown()
    return total_memory

if __name__ == "__main__":
    total_memory = get_total_gpu_memory()
    print(f"Total GPU Memory: {total_memory / 1024**2} MB")
