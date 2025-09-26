import torch
print('Torch バージョン:', torch.__version__)
print('GPUが使えるか:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPUの名前:', torch.cuda.get_device_name(0))

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
print('語彙サイズ:', tokenizer.vocab_size)

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
print('モデルを読み込みました')

from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./gpt2_wikipedia',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,     # まずは1エポック
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


N = 19913
for n_dataset in range(N):
    with open(f"data/data_text_wikipedia_{n_dataset:05}") as f:
        raw = f.read()
    encodings = tokenizer(raw, return_tensors='pt', add_special_tokens=False)
    input_ids = encodings['input_ids'][0]

    block_size = 128
    examples = []
    for i in range(0, input_ids.size(0) - block_size + 1, block_size):
        block = input_ids[i:i+block_size]
        examples.append({'input_ids': block.tolist(), 'attention_mask': [1]*block_size})

    dataset = Dataset.from_list(examples)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    print(f"dataset: {n_dataset}")
    prompt = "私は"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    out = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
    print("生成された文章:")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    model.save_pretrained("gpt2_wikipedia/backup")
    if n_dataset % 10 == 9:
        trainer.push_to_hub(f"dataset number {n_dataset}")

# ==============================
# ✅ まとめ
# - Lossが下がることを見て学習が進む実感を得られる
# - 学習後に日本語っぽい文章が出てくる
# - 小さな「事前学習体験」をスマホからでも実行できる！
