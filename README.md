# LLM-PE-Test

# PS

由于手头测试的显存不够，所以把代码中的“upcast attention to fp32”修改为了fp16，根据一些资料，这样做可能会导致精度下降，有需要可以改回来

https://github.com/huggingface/transformers/issues/24519

# Test

args
```
model_path : 路径
txt : 指定测试的文件
pe : pe类型，目前支持rerope/leaky_rerope/ntk_rope_mix/llama_scaling
    来自于https://github.com/bojone/rerope和transformers==4.31.0
alpha : llama_scaling的配套参数，可以是浮点数或者"auto"
temp : 目前两种，llama2(openbuddy) llama2_zh(chinese_llama2)
repeat : 重复生成次数
```

openbuddy-llama2
```shell
python llama2_test.py \
  --model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
  --temp="llama2" \
  --gpus="0,1,2,3" \
  --pe="llama_scaling" \
  --alpha="auto" \
  --txt="data/context.6719.txt" \
  --repeat="2"
```

chinese-llama2
```shell
python llama2_test.py \
  --model_path="models/chinese-alpaca-2-7b-hf" \
  --temp="llama2_zh" \
  --gpus="0,1,2,3" \
  --pe="rerope" \
  --txt="data/context.6719.txt" \
  --repeat="1"
```

# Reference

https://github.com/bojone/rerope

https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

https://github.com/OpenBuddy/OpenBuddy