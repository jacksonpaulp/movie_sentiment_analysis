import torch
from torch.utils.data import Dataset

MAX_LEN = 300

def encode(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

def pad_sequence(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = encode(self.texts.iloc[idx], self.vocab)
        seq = pad_sequence(seq, MAX_LEN)
        return torch.tensor(seq), torch.tensor(self.labels.iloc[idx], dtype=torch.float)
