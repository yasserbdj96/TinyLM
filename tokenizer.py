import pickle
import re

class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, text, max_vocab_size=10000):
        # Tokenize by words and punctuation
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:max_vocab_size - 3]]
        
        # Build vocabulary with special tokens
        special_tokens = ['<PAD>', '<UNK>', '<EOS>']
        vocab = special_tokens + top_words
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
    def encode(self, text):
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
    
    def decode(self, tokens):
        words = [self.idx_to_word.get(t, '<UNK>') for t in tokens]
        # Basic detokenization
        text = ' '.join(words)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.vocab_size = data['vocab_size']