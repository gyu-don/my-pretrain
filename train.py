import torch
from datasets import Dataset, load_dataset
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments
print('Torch バージョン:', torch.__version__)
print('GPUが使えるか:', torch.cuda.is_available())

config = GPT2Config.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel(config)

print(f"{tokenizer.model_max_length = }")


training_args = TrainingArguments(
    output_dir='./gpt2_aozora',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    report_to="none",   # ← wandb を無効化
)

def data_collator(features):
    input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


dataset = load_dataset(
    "globis-university/aozorabunko-clean",
)["train"].filter(
    lambda example: example['meta']['文字遣い種別'] == '新字新仮名',
).map(
    lambda examples: tokenizer(examples["text"], return_attention_mask=True, truncation=True, max_length=tokenizer.model_max_length, padding=True),
    batched=True,
    remove_columns=["text", "meta", "footnote"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()

prompt = "私は"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
out = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
print("生成された文章:")
print(tokenizer.decode(out[0], skip_special_tokens=True))
model.push_to_hub(repo_id="gyu-don/gpt2_aozora")
