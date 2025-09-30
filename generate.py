from transformers import GPT2TokenizerFast, GPT2LMHeadModel
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#model = GPT2LMHeadModel.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./gpt2_wikipedia")

prompt = "私は"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
for i in range(5):
    out = model.generate(input_ids, max_length=200, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
