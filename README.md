# DNA Sequences Generation using SeqGAN

This project implements a **Sequence Generative Adversarial Network (SeqGAN)** to generate synthetic DNA sequences of length 56. It uses an LSTM-based Generator and a CNN-based Discriminator, optimized via Reinforcement Learning (REINFORCE) with Monte Carlo rollouts.

## Features
- **MLE Pretraining:** Generator is pretrained on real sequences to stabilize training.
- **Adversarial Training:** Uses the Discriminator as a reward signal for the Generator.
- **One-Hot Encoding:** Maps nucleotides (A, C, G, T) into numerical tensors.

## Getting Started
1. Clone the repo: `git clone https://github.com/your-username/DNA-SeqGAN.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Place your data in `data/data.csv`.
4. Run training: `python train.py`

## Results
The model generates 56-length nucleotide strings that mimic the distribution of the training set.
