import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import numpy as np

# Set device (Will use GPU/DirectML if configured, otherwise falls back to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Single-layer LSTM with hidden size 128.
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        # Includes dropout and a linear output layer.
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)

def build_vocab(texts, vocab_size=20000):
    # Word-level vocabulary limited to 20k tokens, built only from the training split to avoid test leakage.
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(vocab_size - 1)
    vocab = {word: i + 1 for i, (word, _) in enumerate(most_common)}
    return vocab

def encode_and_pad(texts, vocab, max_len=200):
    # Sequences are pre-padded so the actual text sits at the end of the tensor.
    encoded = []
    for text in texts:
        tokens = [vocab.get(word, 0) for word in text.split()]
        if len(tokens) < max_len:
            # FIX: Add padding zeros to the FRONT, not the back
            tokens = [0] * (max_len - len(tokens)) + tokens
        else:
            # FIX: If truncating, keep the END of the review (where conclusions are)
            tokens = tokens[-max_len:]
        encoded.append(tokens)
    return np.array(encoded)

def train_lstm():
    print(f"Training LSTM on device: {device}")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    vocab = build_vocab(train_df['review'].values)
    
    X_train = encode_and_pad(train_df['review'].values, vocab)
    X_test = encode_and_pad(test_df['review'].values, vocab)
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = LSTMClassifier(vocab_size=20000, embed_dim=100, hidden_size=128).to(device)
    criterion = nn.BCELoss()
    # Model is trained with the Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trained for 5 epochs.
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # It is expected to land somewhat below the classical TF-IDF models, which aligns with findings from a 2026 comparative paper.
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.long).to(device)
        test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)
        predictions = model(test_inputs).squeeze()
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == test_labels).float()
        acc = correct.sum() / len(correct)
        print(f"\nLSTM Test Accuracy: {acc.item()*100:.2f}%")

if __name__ == "__main__":
    train_lstm()