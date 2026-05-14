import regex as re
import ast
from typing import Iterable, Iterator


# Module-level constants
_PATTERN_VOCAB = r'[:\s]+'
_BYTE_LITERAL_PATTERN = r"b'[^']*'|b\"[^\"]*\""

class Tokenizer:
   
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges 
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []

        # for faster lookups
        self.merges_in_priority: dict[tuple, int] = {}
        self.bytes_to_token = {}
        self.special_token_ids = {}
        
        for k,v in self.vocab.items():
            self.bytes_to_token[v] = k

        for i in range(0, len(self.merges)):
            self.merges_in_priority[self.merges[i]] = i
        
        for special_token in self.special_tokens:
            self.special_token_ids[vocab[special_token]] = vocab[special_token]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        local_vocab = {}
        local_merge = []
        
        # vocab
        with open(vocab_filepath, "r", encoding = "utf-8") as v:
            for voc in v:
                
                result = re.split(_PATTERN_VOCAB, voc, maxsplit=1)
                local_vocab[int(result[0])] = ast.literal_eval(result[1].strip())
        
        # merge
        with open(merges_filepath, "r", encoding= "utf-8") as m:
            for voc in m:
                matches = re.findall(_BYTE_LITERAL_PATTERN, voc)
                if len(matches) == 2:
                    pair = tuple(ast.literal_eval(m) for m in matches)
                local_merge.append(pair)

        return cls(vocab=local_vocab, merges=local_merge, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        parts = [bytes([b]) for b in text.encode("utf-8")]

        if not parts:
            return []

        while len(parts) > 1:
            best_pair = None
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merges_in_priority:
                    if best_pair is None or self.merges_in_priority[pair] < self.merges_in_priority[best_pair]:
                        best_pair = pair

            if best_pair is None:
                break

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    # Concatenate the bytes: b'a' + b'b' -> b'ab'
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return [self.bytes_to_token[item] for item in parts]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        special_pattern = f"({'|'.join(re.escape(t) for t in self.special_tokens)})"

        for chunk in iterable:
            buffer += chunk

            parts = re.split(special_pattern, buffer)

            for i in range(len(parts) - 1):
                segment = parts[i]
                if not segment:
                    continue
                    
                if segment in self.special_tokens:
                    yield self.special_token_to_id[segment]
                else:
                    yield from self.encode(segment)
            
            buffer = parts[-1]

        if buffer:
            if buffer in self.special_tokens:
                yield self.special_token_to_id[buffer]
            else:
                yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        str = ""
        for val in ids:
            str = str + self.vocab[val].decode("utf-8")
        
        return str

    