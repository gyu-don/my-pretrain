from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print('語彙サイズ:', tokenizer.vocab_size)

# ==============================
# セル 5: データを数字化
# ==============================

# ==============================
# セル 6: モデルを読み込み
# ==============================
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gyu-don/gpt2_aozora")
print('モデルを読み込みました')

# ==============================
# セル 9: テキスト生成
# ==============================
prompt = "私は"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print("生成された文章:")
for i in range(5):
    out = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

# ==============================
# ✅ まとめ
# - Lossが下がることを見て学習が進む実感を得られる
# - 学習後に日本語っぽい文章が出てくる
# - 小さな「事前学習体験」をスマホからでも実行できる！
