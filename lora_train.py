import torch
from datasets import Dataset, load_dataset
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
print('Torch バージョン:', torch.__version__)
print('GPUが使えるか:', torch.cuda.is_available())

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_name = "gpt2_wikipedia"

orig_model = GPT2LMHeadModel.from_pretrained(f"./{model_name}")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"],
    bias="none",
)

model = get_peft_model(orig_model, lora_config)
model.print_trainable_parameters()

train_data = "aozora"

output_dir = f"./{model_name}_{train_data}"
if train_data == "aozora":
    dataset = load_dataset(
        "globis-university/aozorabunko-clean",
    )["train"].filter(
        lambda example: example['meta']['文字遣い種別'] == '新字新仮名',
    )
    repo_id = f"gyu-don/{model_name}_aozora"
    remove_columns = ["text", "meta", "footnote"]
elif train_data == "wikipedia":
    dataset = load_dataset("llm-book/japanese-wikipedia")["train"]
    repo_id = f"gyu-don/{model_name}_wikipedia"
    remove_columns = ["text", "meta"]
else:
    raise ValueError(f"unknown {train_data = }")


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=3e-4,
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


train_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], return_attention_mask=True, truncation=True, max_length=tokenizer.model_max_length, padding=True),
    batched=True,
    remove_columns=remove_columns,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
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
model.push_to_hub(repo_id=repo_id)
