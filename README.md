# LLM-PE-Test

# PS

由于手头测试的显存不够，所以把代码中的“upcast attention to fp32”修改为了fp16，根据一些资料，这样做可能会导致精度下降，有需要可以改回来

https://github.com/huggingface/transformers/issues/24519

# Test

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