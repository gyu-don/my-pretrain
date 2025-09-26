# 📘 Colab Notebook: GPT-2 small を青空文庫で学ぶ「初心者向け教材」版（修正版）
# スマホでもコピペ1回で動かせるように、全部まとめています。

# ==============================
# セル 1: 必要なライブラリをインストール
# ==============================

# ==============================
# セル 2: GPU 確認
# ==============================
import torch
print('Torch バージョン:', torch.__version__)
print('GPUが使えるか:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPUの名前:', torch.cuda.get_device_name(0))

# ==============================
# セル 3: 青空文庫からテキスト取得
# ==============================
import requests, re
url = "https://www.aozora.gr.jp/cards/001235/files/49866_41897.html"
resp = requests.get(url)
resp.encoding = resp.apparent_encoding
html = resp.text
text = re.sub(r'<[^>]+>', '', html)
text = re.sub(r"\r", "", text)
text = re.sub(r"\n{2,}", "\n", text)
open('data/data_text_aozora.txt', 'w', encoding='utf-8').write(text)
print('サンプルテキスト保存。文字数=', len(text))

# ==============================
# セル 4: トークナイザー準備
# ==============================
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
print('語彙サイズ:', tokenizer.vocab_size)

# ==============================
# セル 5: データを数字化
# ==============================
from datasets import Dataset
with open('data/data_text_aozora.txt', 'r', encoding='utf-8') as f:
    raw = f.read()

encodings = tokenizer(raw, return_tensors='pt', add_special_tokens=False, truncation=True, max_length=80000)
input_ids = encodings['input_ids'][0]

block_size = 128
examples = []
for i in range(0, input_ids.size(0) - block_size + 1, block_size):
    block = input_ids[i:i+block_size]
    examples.append({'input_ids': block.tolist(), 'attention_mask': [1]*block_size})

print('かたまり数:', len(examples))
dataset = Dataset.from_list(examples)

# ==============================
# セル 6: モデルを読み込み
# ==============================
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
print('モデルを読み込みました')

# ==============================
# セル 7: 学習設定
# ==============================
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
    import torch
    input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ==============================
# セル 8: 学習実行
# ==============================
print("学習を始めます...")
trainer.train()

# ==============================
# セル 9: テキスト生成
# ==============================
prompt = "私は"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
out = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
print("生成された文章:")
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ==============================
# 保存
# ==============================
#save_path = "gpt2_aozora/trained"
#model.save_pretrained(save_path)
#tokenizer.save_pretrained(save_path)
trainer.push_to_hub("first-pretrain")
tokenizer.push_to_hub("first-pretrain")

# ==============================
# ✅ まとめ
# - Lossが下がることを見て学習が進む実感を得られる
# - 学習後に日本語っぽい文章が出てくる
# - 小さな「事前学習体験」をスマホからでも実行できる！
