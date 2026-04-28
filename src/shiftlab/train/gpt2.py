from transformers import AutoModelForCausalLM 

MODEL_NAME = "gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(type(model))
