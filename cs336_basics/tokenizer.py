import regex as re
import ast


# Module-level constants
_PATTERN_VOCAB = r'[:\s]+'
_PATTERN_MERGE = r"(b'.*?')"

class Tokenizer:
   
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges 
        self.special_tokens = special_tokens

        # for faster lookups
        self.merges_in_priority: dict[tuple, int] = {}
        self.bytes_to_token = {}
        
        for k,v in self.vocab.items():
            self.bytes_to_token[v] = k

        for i in range(0, len(self.merges)):
            self.merges_in_priority[self.merges[i]] = i

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
                result = re.split(_PATTERN_MERGE, voc, maxsplit=1)
                local_merge.append((ast.literal_eval(result[1]), ast.literal_eval(result[2])))

        return cls(vocab=local_vocab, merges=local_merge, special_tokens=special_tokens)
    
    # byte level encoder
    def encode(self, text: str) -> list[int]:
        encoded_text = []
        for iter in text:
            for byte in iter.encode("utf-8"):
                encoded_text.append(self.bytes_to_token[byte])
        return encoded_text

    
    # def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    #     NotImplemented

    # def decode(self, ids: list[int]) -> str:
    #     NotImplemented

    