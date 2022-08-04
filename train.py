import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import GeneratorNetwork, DiscriminatorNetwork, VGGLoss
from dataset import ImageDataset


if __name__ == "__main__":

    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageDataset("data/train")
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)

    generator = GeneratorNetwork(in_channels=3, num_channels=64, num_blocks=16).to(device)
    discriminator = DiscriminatorNetwork(in_channels=3, num_blocks=8).to(device)
    
    generator_optim = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    vgg_loss = VGGLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):

        data = tqdm(loader, leave=True)

        for i, (low_resolution, high_resolution) in enumerate(data):

            # move data to device
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)

            # discriminator step
            generated = generator(low_resolution)
            real = discriminator(high_resolution)
            fake = discriminator(generated.detach())

            discriminator_real_loss = bce_loss(real, torch.ones_like(real) - 0.1 * torch.rand_like(real))
            discriminator_fake_loss = bce_loss(fake, torch.zeros_like(fake))
            discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss

            discriminator_optim.zero_grad()
            discriminator_total_loss.backward()
            discriminator_optim.step()

            # generator step
            fake = discriminator(generated)
            adversarial_loss_score = 0.001 * bce_loss(fake, torch.ones_like(fake))
            vgg_loss_score = 0.006 * vgg_loss(generated, high_resolution)
            generator_total_loss = adversarial_loss_score + vgg_loss_score

            generator_optim.zero_grad()
            generator_total_loss.backward()
            generator_optim.step()

            if i % 100 == 0:

                save_image(generated * 0.5 + 0.5, f"test/epoch_{epoch}_iter_{i}.png")