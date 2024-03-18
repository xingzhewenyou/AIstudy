#尝试下载模型https://huggingface.co/LinkSoul/Chinese-Llama-2-7b

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import sentencepiece

# 输出模型配置信息


model_path = "LinkSoul/Chinese-Llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

prompt = instruction.format("用英文回答，什么是夫妻肺片？")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
