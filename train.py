import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import GeneratorNetwork, DiscriminatorNetwork, VGGLoss
from dataset import ImageDataset, test_transform, process_lowres
from PIL import Image

def save_checkpoint(epoch, model, optimizer, filename):
    print("***** Saving checkpoint *****")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("***** Loading checkpoint *****")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    epoch = checkpoint["epoch"]
    print(f"Checkpoint from epoch={epoch}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return checkpoint

def test_inference(test_path, output_path, epoch, generator, device):

    generator.eval()
    with torch.no_grad():
        files = os.listdir(test_path)

        for file in files:
            image = Image.open(test_path + file)

            low_res = process_lowres(image=np.asarray(image))["image"].unsqueeze(0).to(device)
            upscaled_img = generator(low_res)
            save_image(upscaled_img * 0.5 + 0.5, f"{output_path}epoch_{epoch}_{file}")
    generator.train()


def train(loader, generator, discriminator, generator_optim, discriminator_optim, mse_loss, bce_loss, vgg_loss):
    
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
        discriminator_total_loss = discriminator_fake_loss + discriminator_real_loss

        discriminator_optim.zero_grad()
        discriminator_total_loss.backward()
        discriminator_optim.step()

        # generator step
        fake = discriminator(generated)
        l2_loss = mse_loss(generated, high_resolution)
        adversarial_loss_score = 0.001 * bce_loss(fake, torch.ones_like(fake))
        vgg_loss_score = 0.006 * vgg_loss(generated, high_resolution)
        generator_total_loss =  vgg_loss_score + adversarial_loss_score + l2_loss

        generator_optim.zero_grad()
        generator_total_loss.backward()
        generator_optim.step()

        print(f"epoch={epoch}, iteration={i}, gen_loss={generator_total_loss}, disc_loss={discriminator_total_loss}")

        if i == 100:
            save_image(generated * 0.5 + 0.5, f"data/output/epoch_{epoch}_training_gen.png")
            save_image(high_resolution * 0.5 + 0.5, f"data/output/epoch_{epoch}_training_real.png")

if __name__ == "__main__":

    NUM_EPOCHS = 500
    LEARNING_RATE = 0.00001 #0.0001 changed learning rate for later epochs >350
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    CHECKPOINT_GEN = "checkpoints/gen.pth.tar"
    CHECKPOINT_DISC = "checkpoints/disc.pth.tar"

    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageDataset("data/train")
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

    generator = GeneratorNetwork(in_channels=3, num_channels=64, num_blocks=16).to(device)
    discriminator = DiscriminatorNetwork(in_channels=3).to(device)
    
    generator_optim = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    vgg_loss = VGGLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    starting_epoch = 0
    # Load checkpoint
    if os.path.exists(CHECKPOINT_GEN) and os.path.exists(CHECKPOINT_DISC):
        gen_checkpoint = load_checkpoint(CHECKPOINT_GEN, generator, generator_optim, LEARNING_RATE, device)
        disc_checkpoint = load_checkpoint(CHECKPOINT_DISC, discriminator, discriminator_optim, LEARNING_RATE, device)
        starting_epoch = gen_checkpoint["epoch"]

    # TODO: Make range start from checkpoint epoch if loaded
    for epoch in range(starting_epoch, NUM_EPOCHS):

        train(loader=loader, generator=generator, discriminator=discriminator, generator_optim=generator_optim, discriminator_optim=discriminator_optim, mse_loss=mse_loss, bce_loss=bce_loss, vgg_loss=vgg_loss)
        
        save_checkpoint(epoch, generator, generator_optim, CHECKPOINT_GEN)
        save_checkpoint(epoch, discriminator, discriminator_optim, CHECKPOINT_DISC)

        #if epoch % 5 == 0:
        #    test_inference(test_path="data/test/", output_path="data/output/", epoch=epoch, generator=generator, device=device)