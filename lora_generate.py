from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from peft import PeftConfig, PeftModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

lora_name = "gpt2_wikipedia_aozora"

config = PeftConfig.from_pretrained(f"gyu-don/{lora_name}")
orig_model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(orig_model, lora_name)

prompt = "牛丼"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
for i in range(5):
    out = model.generate(input_ids, max_length=200, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
