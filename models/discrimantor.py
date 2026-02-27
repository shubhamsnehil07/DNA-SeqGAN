import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, vocab_size=4, seq_length=56):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(vocab_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Based on 56 length: 56 -> 28 (pool1) -> 14 (pool2)
        self.fc1 = nn.Linear(128 * 14, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, vocab) -> (batch, vocab, seq_len)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
