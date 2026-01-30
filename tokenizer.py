class Tokenizer: # Simple BPE Tokenizer
    def __init__(self, source: str, source_is_trained: bool = False, iterations: int = 10, end_of_word: str = '</w>', unrecognized: str = '</u>'):
        self.eow: str = end_of_word
        self.urc: str = unrecognized
        self.merge_table: list[str] = []
        self.vocab: dict[str, int] = {self.eow: 0}

        if source_is_trained:
            self.load(source) 
        else:
            self.train_large(source, iterations)

    def reset(self):
        self.merge_table.clear()
        self.vocab.clear()
        self.vocab = {self.eow: 0}

    def process_string(self, data) -> list[str]:
        lines = data.split("\n")
        words: list[str] = []
        for line in lines:
            words.extend(line.split(" "))

        return words

    def train_large(self, source: str, iterations: int = 10, freq_threshold: int = 10):
        self.reset()
        for _ in range(iterations):        
            word_freq: dict[list[int], int] = {}
            pair_freq: dict[list[int], int] = {}
            best_pair: list[int] | None = None

            with open(source, 'r', encoding='utf-8') as file:
                for line in file:
                    # Process line into a list of words
                    words: list[str] = self.process_string(line)

                    # Update base vocab
                    for word in words:
                        for c in word:
                            if c not in self.vocab:
                                self.vocab[c] = len(self.vocab)
                        
                        # Update word frequencies
                        word_freq[self.tokenize(word)[0]] = word_freq.get(self.tokenize(word)[0], 0) + 1

                    # Find best token pair
                    for word, freq in word_freq.items():
                        for i in range(len(word) - 1):
                            pair = [word[i]] + [word[i+1]]
                            pair_freq[pair] = pair_freq.get(pair, 0) + freq

                            if best_pair is None or pair_freq[pair] > pair_freq.get(best_pair, 0):
                                best_pair = pair

                # Exit early if no best_pair is found
                if best_pair is None:
                    return

                # Exit early if best_pair is does not pass frequency threshold
                if pair_freq.get(best_pair, 0) < freq_threshold:
                    return

                # Add best pair as a new token
                best_pair_stringified = self.stringify([best_pair])[0]
                best_pair_string = best_pair_stringified[0] + best_pair_stringified[1]
                
                self.vocab[best_pair_string] = len(self.vocab)
                self.merge_table.append(best_pair_string)

                # Replace all instances of pair with new token
                for word, freq in word_freq.items():
                    i = 0
                    while i < len(word) - 1:
                        if word[i] + word[i+1] == best_pair:
                            word[i] = self.tokenize(best_pair_string)[0][0]
                            word.pop(i+1)
                        else:
                            i += 1
                            
    def train_small(self, source: str, iterations: int = 10, freq_threshold: int = 10):
        # Read data
        data: str = ""
        with open(source, 'r', encoding='utf-8') as file:
            data = file.read()

        # Process data into words
        words = self.process_string(data)
        
        # Initialize vocab
        self.vocab = {self.eow: 0}
        for word in words:
            for c in word:
                if c not in self.vocab:
                    self.vocab[c] = len(self.vocab)

        # Process words into initial tokens
        tokenized_words: list[list[str]] = []
        for word in words:
            tokenized_words.append(list(word) + [self.eow])

        # Repeatedly merge most frequent token pairs
        for _ in range(iterations):

            # Find best token pair
            best_pair: str | None = None
            pair_freq: dict[str, int] = {}
            for word in tokenized_words:
                for i in range(len(word) - 1):
                    pair = word[i] + word[i+1]
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1

                    if best_pair is None or pair_freq[pair] > pair_freq.get(best_pair, 0):
                        best_pair = pair

            # Exit early if no best_pair is found
            if best_pair is None:
                return

            # Exit early if best_pair is does not pass frequency threshold
            if pair_freq.get(best_pair, 0) < freq_threshold:
                return

            # Add best pair as a new token
            self.vocab[best_pair] = len(self.vocab)
            self.merge_table.append(best_pair)

            # Replaces all instances of pair with new token
            for word in tokenized_words:
                i = 0
                while i < len(word) - 1:
                    if word[i] + word[i+1] == best_pair:
                        word[i] = best_pair
                        word.pop(i+1)
                    else:
                        i += 1

    def tokenize(self, input: str) -> list[list[int]]:
        # Process string into words
        words = self.process_string(input)

        result: list[list[int]] = []
        # Tokenize based on merge table
        for word in words:
            tokens = list(word) + [self.eow]
            for merged_pair in self.merge_table:
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] + tokens[i+1] == merged_pair:
                        tokens[i] = merged_pair
                        tokens.pop(i+1)
                    else:
                        i += 1

            result.append([self.vocab.get(token, -1) for token in tokens]) #-1 = self.urc

        return result
    
    def stringify(self, input: list[list[int]]) -> list[list[str]]:
        result: list[list[str]] = []

        reverse_vocab = {v: k for k, v in self.vocab.items()}

        for token_group in input:
            word: list[str] = []
            for token in token_group:
                word.append(reverse_vocab.get(token, self.urc))
            result.append(word)

        return result

    def save(self, output: str) -> bool:
        return True

    def load(self, filepath: str) -> bool:
        return True
    
tokenizer = Tokenizer(
    './data/tokenizer/book_corpus_small.txt',
    False,
    1
)

while True:
    string = input("Give a string to tokenize:\n\t")
    tokenized = tokenizer.tokenize(string)
    stringified = tokenizer.stringify(tokenized)
    print(f"\n\nTokenized Output:\n\t{tokenized}")
    print(f"\n\nString Representation:\n\t{stringified}\n-----------------------------------")