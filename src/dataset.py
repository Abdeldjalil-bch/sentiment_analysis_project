# ======================
#  CLASSES PYTORCH
# ======================
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import random

class Vocabulary:
    def __init__(self, max_size=15000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<NUM>'}
        self.word_freq = Counter()
    
    def build_vocab(self, texts):
        for text in texts:
            tokens = text.split()
            self.word_freq.update(tokens)
        
        most_common = self.word_freq.most_common(self.max_size - len(self.word2idx))
        for word, freq in most_common:
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text, max_length):
        tokens = text.split()
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens[:max_length]]
        if len(indices) < max_length:
            indices += [self.word2idx['<PAD>']] * (max_length - len(indices))
        return indices
    
    def __len__(self):
        return len(self.word2idx)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.vocab.encode(text, self.max_length)
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != self.vocab.word2idx['<PAD>'] else 0 for i in indices], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(df, max_length=60, batch_size=64, test_size=0.2, random_state=42):
    # Split train/test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']  # Préserver la distribution des classes
    )
    
    # Construire vocabulaire sur le train uniquement
    vocab = Vocabulary(max_size=15000, min_freq=2)
    vocab.build_vocab(train_df['text_clean'].tolist())
    
    print(f"\n📚 Vocabulaire construit : {len(vocab)} mots uniques")
    
    # Créer datasets
    train_dataset = SentimentDataset(
        texts=train_df['text_clean'].tolist(),
        labels=train_df['label'].tolist(),
        vocab=vocab,
        max_length=max_length
    )
    
    test_dataset = SentimentDataset(
        texts=test_df['text_clean'].tolist(),
        labels=test_df['label'].tolist(),
        vocab=vocab,
        max_length=max_length
    )
    
    # Créer dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📦 Dataloaders créés :")
    print(f"   - Train : {len(train_loader)} batches")
    print(f"   - Test  : {len(test_loader)} batches")
    
    return train_loader, test_loader, vocab, train_df, test_df
