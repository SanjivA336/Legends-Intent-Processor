import time
import json
import csv

class Tokenizer: # BPE Tokenizer
    def __init__(self):
        self.eow = "</w>"
        self.urc = "�"
        self.vocab: dict[str, int] = {self.urc: 0, self.eow: 1}
        self.merge_table: list[tuple[str, str]] = []

    def train(self, corpus_path, merges=1000, debug=True):
        start_time = time.time()

        if debug:
            print(f"Starting training at {start_time}...")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Preprocess line
                words = line.removesuffix("\n").strip().split(" ")

                for word in words:
                    # Tokenize word into characters + eow
                    tokens = list(word) + [self.eow]

                    # Build initial vocab with single characters
                    for token in tokens:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)

            if debug:
                print(f"Starting Merges...\nTime Elapsed: {time.time() - start_time:.2f}s")

            first_elapsed: float | None = None
            for m in range(merges):
                iteration_start_time = time.time()
                best_pair: tuple[str, str] | None = None
                self.pair_freq: dict[tuple[str, str], int] = {}
                f.seek(0)
                for line in f:
                    # Preprocess line
                    words = line.removesuffix("\n").strip().split(" ")

                    for word in words:
                        # Retokenize words based on current merge table
                        tokens = self.tokenize_word(word)

                        # Count initial pair frequencies + find best pair
                        for i in range(len(tokens) - 1):
                            if tokens[i] == self.eow or tokens[i + 1] == self.eow:
                                continue

                            pair = (tokens[i], tokens[i + 1])
                            self.pair_freq[pair] = self.pair_freq.get(pair, 0) + 1
                            if best_pair is None or self.pair_freq[pair] > self.pair_freq.get(best_pair, 0):
                                best_pair = pair

                # Early Quit: No more pairs to merge
                if best_pair is None:
                    break

                # Early Quit: Best pair frequency is less than 2
                if self.pair_freq[best_pair] < 2:
                    break

                # Merge best pair into vocab and update merge table if it isn't already present
                if best_pair[0] + best_pair[1] not in self.vocab:
                    self.vocab[best_pair[0] + best_pair[1]] = len(self.vocab)
                    self.merge_table.append(best_pair)

                if debug:
                    elapsed = time.time() - iteration_start_time
                    remaining = 0.0
                    if first_elapsed is None:
                        first_elapsed = elapsed
                    else:
                        total_increase = elapsed - first_elapsed
                        growth_rate = total_increase / (m + 1) # Assume linear growth
                        remaining = (merges - (m + 1)) * (first_elapsed + growth_rate * (merges + m) / 2)


                    print(f"Merge #{m} Time\t->\tIteration Elapsed: {elapsed:.2f}s\tTotal Elapsed : {time.time() - start_time:.2f}s\tRemaining: {remaining:.2f}s\tBest Pair: {best_pair} -> \'{best_pair[0] + best_pair[1]}\'")

    def tokenize_word(self, word: str) -> list[str]:
        tokens = list(word) + [self.eow]

        # Naive tokenization. No greedy matching.
        for merge in self.merge_table:
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair == merge:
                    tokens[i] = pair[0] + pair[1]
                    tokens.pop(i + 1)
                else:
                    i += 1

        return tokens
    
    def tokenize_line(self, line: str) -> list[str]:
        words = line.removesuffix("\n").strip().split(" ")
        tokenized_line = []
        for word in words:
            tokenized_line.extend(self.tokenize_word(word))
        return tokenized_line
    
    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(token, -1) for token in tokens]
    
    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return [inv_vocab.get(id_, self.urc) for id_ in ids]
    
    def save_vocab(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def save_merges(self, path: str):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["left", "right"])
            for left, right in self.merge_table:
                writer.writerow([left, right])

    def save(self, root_path: str):
        self.save_vocab(f"{root_path}_vocab.json")
        self.save_merges(f"{root_path}_merges.csv")

    def load_vocab(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def load_merges(self, path: str):
        import csv
        self.merge_table = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.merge_table.append((row["left"], row["right"]))

    def load(self, root_path: str):
        self.load_vocab(f"{root_path}_vocab.json")
        self.load_merges(f"{root_path}_merges.csv")



bpe = Tokenizer()
bpe.train('data/corpus_sm.txt', 100, debug=True)
bpe.save_merges('saves/bpe_merges.csv')
bpe.save_vocab('saves/bpe_vocab.json')
tokenized = bpe.tokenize_line("hello world")
print("Tokenized:", tokenized)
ids = bpe.tokens_to_ids(tokenized)
print("Token IDs:", ids)
decoded_tokens = bpe.ids_to_tokens(ids)
print("Decoded Tokens:", decoded_tokens)

bpe2 = Tokenizer()
bpe2.load_vocab('saves/bpe_vocab.json')
bpe2.load_merges('saves/bpe_merges.csv')
tokenized2 = bpe2.tokenize_line("hello world")
print("Tokenized after loading:", tokenized2)
ids2 = bpe2.tokens_to_ids(tokenized2)
print("Token IDs after loading:", ids2)
decoded_tokens2 = bpe2.ids_to_tokens(ids2)
print("Decoded Tokens after loading:", decoded_tokens2)
