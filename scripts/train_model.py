import torch
import torch.nn as nn

class SignTranslationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, vocab_size=5000):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out