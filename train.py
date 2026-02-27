import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split
from utils.data_loader import load_dna_data, SEQ_LENGTH, VOCAB_SIZE
from models.generator import Generator
from models.discriminator import Discriminator
from utils.rollout import rollout

# Constants
BATCH_SIZE = 32
G_PRETRAIN_EPOCHS = 50
D_PRETRAIN_EPOCHS = 50
NUM_EPOCHS = 100

def train():
    data = load_dna_data('data/data.csv')
    train_data, _ = train_test_split(data, test_size=0.2, random_state=42)

    generator = Generator()
    discriminator = Discriminator()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    criterion = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    # --- Generator Pretraining ---
    print("Pretraining Generator...")
    for epoch in range(G_PRETRAIN_EPOCHS):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            g_optimizer.zero_grad()
            input_seq = batch[:, :-1]
            target_indices = torch.argmax(batch[:, 1:], dim=-1)
            probs, _ = generator(input_seq)
            loss = ce_loss(probs.view(-1, VOCAB_SIZE), target_indices.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_optimizer.step()

    # --- Discriminator Pretraining ---
    print("Pretraining Discriminator...")
    for epoch in range(D_PRETRAIN_EPOCHS):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            d_optimizer.zero_grad()
            # Real/Fake labels and logic from notebook...
            # (Truncated for brevity, mirrors notebook code)
            d_optimizer.step()

    # --- Adversarial Training ---
    print("Starting Adversarial Training...")
    for epoch in range(NUM_EPOCHS):
        # Implementation of REINFORCE loop...
        pass

    torch.save(generator.state_dict(), 'weights/generator_state_dict.pth')

if __name__ == "__main__":
    train()
