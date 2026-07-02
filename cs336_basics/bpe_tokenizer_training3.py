from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

## USING THIS FOR owT FILES BECAUSE, I AM KEEP GETTING OOM ERROR ON MY MACHINE.
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>"]
)

tokenizer.train(files=["data/owt_train.txt"], trainer=trainer)
tokenizer.save("data/bpe_tokenizer_owt_train_300626.json")

vocab = tokenizer.get_vocab()  # dict {token_str: id}

output_dir = "data/training_output_owt"
os.makedirs(output_dir, exist_ok=True)

tokenizer.model.save(output_dir)  # writes vocab.json, merges.txt

temp_merges_path = os.path.join(output_dir, "merges.txt")  # fix: actual filename
merges = []
with open(temp_merges_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            merges.append(line)

with open(os.path.join(output_dir, "vocab_train_300626.txt"), "w", encoding="utf-8") as f:
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{idx}: {token}\n")

with open(os.path.join(output_dir, "merges_train_300626.txt"), "w", encoding="utf-8") as f:  # fix: different filename
    for merge in merges:
        f.write(f"{merge}\n")

if os.path.exists(os.path.join(output_dir, "vocab.json")):
    os.remove(os.path.join(output_dir, "vocab.json"))
if os.path.exists(temp_merges_path):
    os.remove(temp_merges_path)