import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

tokens = re.findall(PAT, "some text that i'll pre-tokenize")
list_bytes = []
for token in tokens:
    list_bytes.append(tuple(token.encode("UTF-8")))
print(list_bytes)

