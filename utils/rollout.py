import torch
from torch.distributions import Categorical

def rollout(generator, discriminator, seq, seq_length=56, vocab_size=4, num_rollouts=10):
    rewards = []
    device = seq.device
    for _ in range(num_rollouts):
        current_seq = seq.clone()
        hidden = None
        for t in range(seq_length):
            if t < len(seq[0]):
                input_t = torch.zeros(seq.size(0), 1, vocab_size).to(device)
                input_t.scatter_(2, seq[:, t:t+1].unsqueeze(-1), 1)
            else:
                input_t = torch.zeros(seq.size(0), 1, vocab_size).to(device)
                input_t[:, :, 0] = 1  # Default to 'A'
            probs, hidden = generator(input_t, hidden)
            dist = Categorical(probs.squeeze(1))
            next_nucleotide = dist.sample().unsqueeze(1)
            current_seq = torch.cat([current_seq[:, :t], next_nucleotide], dim=1)
            
        one_hot = torch.zeros(seq.size(0), seq_length, vocab_size).to(device)
        one_hot.scatter_(2, current_seq.unsqueeze(-1), 1)
        reward = discriminator(one_hot).detach()
        rewards.append(reward)
    return torch.mean(torch.stack(rewards), dim=0)
