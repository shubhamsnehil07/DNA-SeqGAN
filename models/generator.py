import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, vocab_size=4, hidden_dim=128):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden
