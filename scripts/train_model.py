import torch
import torch.nn as nn
import torch.nn.functional as F

class SignTranslationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, vocab_size=5000, max_seq_len=50):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, frames, features)
        
        # 1. Permute to (batch_size, features, frames) for 1D pooling
        x = x.permute(0, 2, 1) 
        
        # 2. Adaptive Temporal Pooling: Dynamically compress ANY frame count down to exactly 50 steps
        x = F.adaptive_avg_pool1d(x, self.max_seq_len)
        
        # 3. Permute back to (batch_size, 50, features) for the LSTM
        x = x.permute(0, 2, 1)
        
        # 4. Process the aligned sequence
        out, _ = self.lstm(x) # out shape: (batch_size, 50, hidden_dim*2)
        out = self.fc(out)    # out shape: (batch_size, 50, vocab_size)
        
        return out