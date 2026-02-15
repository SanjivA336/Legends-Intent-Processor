from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from tokenizer import Tokenizer

class TokenPredictor:
    def __init__(self, tokenizer, ngram_size=2, topK=5, temperature=1.0):
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        self.topK = topK
        self.temperature = temperature
        self.pair_probs: list[list[float]] = [[0.0 for _ in range(self.tokenizer.vocab_size())] for _ in range(self.tokenizer.vocab_size())]
        
    def train(self, corpus_path, debug=True):
        import time
        start_time = time.time()

        counts: list[list[int]] = [[0 for _ in range(self.tokenizer.vocab_size() + 1)] for _ in range(self.tokenizer.vocab_size())]

        merge_start_time = time.time()

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.lower().removesuffix("\n").strip().split(" ")
                for word in words:
                    tokens = self.tokenizer.tokenize_word(word)
                    # Count pairs
                    for i in range(len(tokens) - 1):
                        counts[self.tokenizer.token_to_id(tokens[i])][self.tokenizer.token_to_id(tokens[i+1])] += 1
                        counts[self.tokenizer.token_to_id(tokens[i])][-1] += 1  # Total count for token i

        # Normalize counts to probabilities
        for i in range(len(counts)):
            for j in range(len(counts)):
                self.pair_probs[i][j] = counts[i][j] / counts[i][-1] if counts[i][-1] > 0 else 0.0

    def predict_next_from_id(self, token_id: int) -> int:
        if token_id < 0 or token_id >= self.tokenizer.vocab_size():
            return 0
        probs = self.pair_probs[token_id]
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        # Apply temperature scaling
        if self.temperature > 0:
            scaled_probs = [pow(p, 1.0 / self.temperature) for p in probs]
            total = sum(scaled_probs)
            if total > 0:
                scaled_probs = [p / total for p in scaled_probs]
            else:
                scaled_probs = [0.0 for _ in scaled_probs]
            sorted_indices = sorted(range(len(scaled_probs)), key=lambda i: scaled_probs[i], reverse=True)
            sorted_indices = sorted_indices[:self.topK]

        return sorted_indices[0] if sorted_indices else 0

    def predict_next_from_token(self, token: str) -> str:
        token_id = self.tokenizer.token_to_id(token)
        next_token_id = self.predict_next_from_id(token_id)
        return self.tokenizer.id_to_token(next_token_id)

    def predict_next_from_word(self, word: str) -> str:
        tokens = self.tokenizer.tokenize_word(word)

        filtered = [t for t in tokens if t != self.tokenizer.eow]

        if not filtered:
            return self.tokenizer.id_to_token(0)

        last_token = filtered[-1]
        print(f"Predicting next token for '{last_token}'...")

        return self.predict_next_from_token(last_token)

    
    def print_probs(self):
        print("Tokens\t|", end="")
        for i in range(self.tokenizer.vocab_size()):
            token = self.tokenizer.id_to_token(i)
            print(f"{token}\t|", end="")
        print("\n", end="")

        for i in range(self.tokenizer.vocab_size()):
            token = self.tokenizer.id_to_token(i)
            print(f"{token}:\t|", end="")
            for j in range(self.tokenizer.vocab_size()):
                next_token = self.tokenizer.id_to_token(j)
                prob = self.pair_probs[i][j]
                print(f"{prob:.2f}\t|", end="")
            print("\n")
    
tokenizer = Tokenizer()
tokenizer.load("saves/100md")
predictor = TokenPredictor(tokenizer)
predictor.train("data/corpus_sm.txt", debug=True)
predictor.print_probs()
while True:
    user_input = input("Enter a token (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    predicted_token = predictor.predict_next_from_token(user_input)
    print(f"Predicted next token (as Token): {predicted_token}")
    predicted_token = predictor.predict_next_from_word(user_input)
    print(f"Predicted next token (as Word): {predicted_token}")