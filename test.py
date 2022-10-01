
import torch
from torch import optim
from train import load_checkpoint, test_inference   
from model import GeneratorNetwork



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 0.0001
    CHECKPOINT_GEN = "checkpoints/gen.pth.tar"

    generator = GeneratorNetwork(in_channels=3, num_channels=64, num_blocks=16).to(device)
    generator_optim = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999))

    checkpoint = load_checkpoint(CHECKPOINT_GEN, generator, generator_optim, LEARNING_RATE, device)

    test_inference("data/test/", "data/output/", checkpoint["epoch"], generator, device)



