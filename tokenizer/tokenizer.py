from collections import defaultdict
from typing import List, Optional, Tuple


import regex as re
from tqdm import tqdm


class BasicTokenizer:
    TOKEN_PATTERN = (r"'(?:s|t|re|ve|m|ll|d)"
                     r"| ?\p{L}+"
                     r"| ?\p{N}+"
                     r"| ?[^\s\p{L}\p{N}]+"
                     r"|\s+(?!\S)|\s+")

    @classmethod
    def tokenize(cls, text: str):
        return re.findall(cls.TOKEN_PATTERN, text)


class Token:
    def __init__(self, token: str, occ: Optional[int] = 1):
        self.name, self.occ = token, occ
        self.ids = list(token.encode('utf-8'))
        self.pairs = list(zip(self.ids[:-1], self.ids[1:]))

    def merge(self, target_pair: Tuple[int], new_token_idx: int) -> List[Tuple[int]]:
        is_there_pair = False
        for i, _ in enumerate(self.pairs):
            pair = self.pairs[i]
            if pair == target_pair:
                is_there_pair = True
                if i - 1 >= 0:
                    old_previous_pair = self.pairs[i-1]
                    previous_pair = (old_previous_pair[0], new_token_idx)
                    self.pairs[i-1] = previous_pair

                if i + 1 < len(self.pairs):
                    old_next_pair = self.pairs[i+1]
                    next_pair = (new_token_idx, old_next_pair[1])
                    self.pairs[i+1] = next_pair

        if is_there_pair:
            self.pairs = [pair for pair in self.pairs if pair != target_pair]

        return self.pairs

    def merge_ids(self, pair: Tuple[int], id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(self.ids):
            if i < len(self.ids) - 1 and self.ids[i] == pair[0] and self.ids[i+1] == pair[1]:
                new_ids.append(id)
                i += 2

            else:
                new_ids.append(self.ids[i])
                i += 1

        self.ids = new_ids
        self.pairs = list(zip(self.ids[:-1], self.ids[1:]))
        return new_ids

    def __len__(self):
        return len(self.ids)


class BPETokenizer:
    def __init__(self, basic_tokenizer=BasicTokenizer):
        self.__basic_tokenizer = BasicTokenizer()

    def train(self, corpus: str, n_merges: int = 100):
        basic_tokens = self.__basic_tokenizer.tokenize(corpus)
        print(f"tokenized {len(basic_tokens):,} basic tokens")

        occ = defaultdict(int)
        for token in set(basic_tokens):
            occ[token] += 1

        print(f'Counted {len(occ) :,} different basic tokens')

        self.vocab = {i: bytearray([i]) for i in range(256)}
        print(f"Initial tokens: {len(self.vocab):,}")
        self.corpus_tokens = [Token(token, occ) for token, occ in occ.items()]

        self.merges = {}
        for _ in tqdm(range(n_merges)):
            counting = defaultdict(int)
            for token in self.corpus_tokens:
                for pair in token.pairs:
                    counting[pair] += token.occ

            if not counting:
                break

            pair, pair_occ = max(counting.items(), key=lambda x: x[1])

            new_idx = len(self.vocab)
            token_a, token_b = pair
            new_token = self.vocab[token_a] + self.vocab[token_b]
            self.vocab[new_idx] = new_token
            self.merges[pair] = new_idx

            for token in self.corpus_tokens:
                token.merge(pair, new_idx)

        self.inverse_vocab = {token.decode('utf-8', errors='replace'): idx for idx, token in self.vocab.items()}
        print(f"Tokens {len(self.vocab):,}")

    def encode(self, text: str) -> List[int]:
        tokens_ids = []
        for basic_token in self.__basic_tokenizer.tokenize(text):
            token = Token(basic_token)

            while len(token) > 1:
                pair = min(token.pairs, key = lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break

                idx = self.merges[pair]
                token.merge_ids(pair, idx)

            tokens_ids.extend(token.ids)

        return tokens_ids

    def convert_ids_to_tokens(self, tokens_ids: List[int]) -> List[str]:
        return [self.vocab[id].decode('utf-8', errors='replace') for id in tokens_ids]

    def decode(self, tokens_ids: List[int]) -> str:
        tokens = b''.join(self.vocab[id] for id in tokens_ids)
        return tokens.decode('utf-8', errors='replace')
