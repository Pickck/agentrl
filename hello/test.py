from vllm import LLM, SamplingParams

# 如果这一步报错，说明 vLLM 根本没安装好 NPU 支持
llm = LLM(model="/mnt/workspace/models/Qwen2.5-0.5B-Instruct", 
          # device="npu", 
          tensor_parallel_size=2) 

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Output: {output.outputs[0].text}")