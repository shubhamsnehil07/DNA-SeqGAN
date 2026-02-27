import torch
from models.generator import Generator
from utils.data_loader import INV_NUCLEOTIDE_MAP, SEQ_LENGTH, VOCAB_SIZE
from torch.distributions import Categorical

def generate(num_samples=10):
    gen = Generator()
    gen.load_state_dict(torch.load('weights/generator_state_dict.pth'))
    gen.eval()
    
    with torch.no_grad():
        noise = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LENGTH)).long()
        seq = torch.zeros(num_samples, SEQ_LENGTH, VOCAB_SIZE)
        seq.scatter_(2, noise.unsqueeze(-1), 1)
        probs, _ = gen(seq)
        gen_seq = Categorical(probs).sample()
        
        for s in gen_seq:
            print("".join([INV_NUCLEOTIDE_MAP[idx.item()] for idx in s]))

if __name__ == "__main__":
    generate()
