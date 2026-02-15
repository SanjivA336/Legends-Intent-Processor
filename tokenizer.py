import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

def count_pairs_worker(chunk, merge_table, eow) -> Counter[tuple[str, str]]:
    local_counter = Counter()
    for line in chunk:
        words = line.lower().removesuffix("\n").strip().split(" ")
        for word in words:
            tokens = list(word) + [eow]
            # Apply merges
            for merge in merge_table:
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i+1]) == merge:
                        tokens[i] = tokens[i] + tokens[i+1]
                        tokens.pop(i+1)
                    else:
                        i += 1
            # Count pairs
            for i in range(len(tokens) - 1):
                if tokens[i] != eow and tokens[i+1] != eow:
                    local_counter[(tokens[i], tokens[i+1])] += 1
    return local_counter

class Tokenizer: # BPE Tokenizer
    def __init__(self):
        self.eow = "</w>"
        self.urc = "�"
        self.vocab: dict[str, int] = {self.urc: 0, self.eow: 1}
        self.inv_vocab: dict[int, str] = {0: self.urc, 1: self.eow}
        self.merge_table: list[tuple[str, str]] = []

    def train(self, corpus_path, merges=1000, chunk_size=10000, max_workers=8, limit=0.001, debug=True):
        import time
        start_time = time.time()

        num_init_tokens = 0
        # Step 1: Build initial vocab sequentially
        if debug:
            print("Building initial vocab...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.lower().removesuffix("\n").strip().split(" ")
                for word in words:
                    tokens = list(word) + [self.eow]
                    num_init_tokens += len(tokens)
                    for token in tokens:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)
                            self.inv_vocab[len(self.inv_vocab)] = token

        if debug:
            print(f"Initial vocab size: {len(self.vocab)}")
            print(f"Initial tokens count: {num_init_tokens}")
            print(f"Time taken for initial vocab: {time.time() - start_time:.2f}s")

        # Step 2: Perform merges
        prev_time = time.time() - start_time
        total_diff = 0
        for m in range(merges):
            merge_start_time = time.time()

            pair_counter: Counter[tuple[str, str]] = Counter()

            # Step 2a: Create a streaming chunk generator
            def chunk_generator():
                chunk = []
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        chunk.append(line)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    if chunk:
                        yield chunk

            # Step 2c: Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(count_pairs_worker, chunk, self.merge_table, self.eow) for chunk in chunk_generator()]
                for future in as_completed(futures):
                    pair_counter.update(future.result())

            # Step 2d: Find the most frequent pair
            if not pair_counter:
                if debug:
                    print("No pairs left to merge. Stopping.")
                break

            best_pair = max(pair_counter.items(), key=lambda kv: kv[1])[0]
            if pair_counter[best_pair] < (num_init_tokens * limit):
                if debug:
                    print(f"Best pair frequency < {num_init_tokens * limit}. Stopping.")
                break

            # Step 2e: Merge the pair
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = merged_token
                self.merge_table.append(best_pair)

            if debug:
                diff = time.time() - merge_start_time
                total_diff += diff
                avg_diff = total_diff / (m + 1)
                x1 = m + 1
                x2 = merges - (m + 1)
                y1 = time.time() - merge_start_time
                y2 = avg_diff * x2 / x1 if x1 > 0 else 0
                print(f"Merge {m+1}/{merges}: {time.time() - start_time:.2f}s\n\t| Best Pair: {best_pair} (Count: {pair_counter[best_pair]})\n\t| Time: {time.time() - merge_start_time:.2f}s\n\t| Difference: {diff:.2f}s\n\t| Avg Difference: {avg_diff:.2f}s\n\t| Est. Time Remaining: {(y2-y1) * (x2-x1) * 0.5:.2f}s")
                prev_time = time.time() - merge_start_time

    def tokenize_word(self, word: str) -> list[str]:
        tokens = list(word.lower()) + [self.eow]

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
        return [self.token_to_id(token) for token in tokens]

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token.lower(), 0)

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token(id) for id in ids]

    def id_to_token(self, id: int) -> str:
        return self.inv_vocab.get(id, self.urc)

    def vocab_size(self) -> int:
        return len(self.vocab)

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
        self.inv_vocab = {id: token for token, id in self.vocab.items()}

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

if __name__ == "__main__":
    import time

    bpe = Tokenizer()
    #bpe.train("data/corpus_sm.txt", merges=1000, chunk_size=20, debug=True)
    bpe.load("saves/100md")

    test_text = "The quick brown fox jumps over the lazy dog."
    tokenized = bpe.tokenize_line(test_text)
    print("Tokenized:", tokenized)

    ids = bpe.tokens_to_ids(tokenized)
    print("Token IDs:", ids)

    decoded_tokens = bpe.ids_to_tokens(ids)
    print("Decoded Tokens:", decoded_tokens)
