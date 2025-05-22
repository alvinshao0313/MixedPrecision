import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B",
                                             torch_dtype="auto",
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
