import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define ESRGAN Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
        )
        
    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.block(x)
        out = self.avg_pool(out)
        return self.sigmoid(out.view(-1, 1))

# Define loss functions
adversarial_loss = nn.BCELoss()
content_loss = nn.MSELoss()

# Load DIV2K dataset
def load_div2k_dataset(dataset_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
    ])
    return datasets.ImageFolder(dataset_dir, transform=transform)

# Training function
def train_esrgan(generator, discriminator, dataloader, num_epochs=100, save_path='generator_model.pth'):
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for i, (lr_imgs, _) in enumerate(dataloader):
            lr_imgs = lr_imgs.cuda()

            print("Min pixel value:", lr_imgs.min().item(), "Max pixel value:", lr_imgs.max().item())
            lr_imgs = lr_imgs.clamp(0, 1)
            batch_size = lr_imgs.size(0)

            # Generate high-res images using the generator
            fake_imgs = generator(lr_imgs).clamp(0, 1)

            # Train the discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1).cuda()
            fake_labels = torch.zeros(batch_size, 1).cuda()

            real_loss = adversarial_loss(discriminator(lr_imgs), real_labels)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()
            g_loss = content_loss(fake_imgs, lr_imgs)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

    # Save the generator model after training
    torch.save(generator.state_dict(), save_path)
    print(f"Generator model saved at {save_path}")

# Evaluation metrics (PSNR, SSIM)
def evaluate_sr_model(generator, dataloader):
    psnr_total = 0
    ssim_total = 0
    for i, (lr_imgs, _) in enumerate(dataloader):
        lr_imgs = lr_imgs.cuda()
        with torch.no_grad():
            sr_imgs = generator(lr_imgs).cpu().numpy()
            lr_imgs = lr_imgs.cpu().numpy()

        for i in range(sr_imgs.shape[0]):
            sr_img = np.clip(sr_imgs[i].transpose(1, 2, 0), 0, 1)
            lr_img = np.clip(lr_imgs[i].transpose(1, 2, 0), 0, 1)

            psnr_total += psnr_metric(lr_img, sr_img)
            ssim_total += ssim_metric(lr_img, sr_img, multichannel=True, win_size=3, data_range=1.0)



    psnr_avg = psnr_total / len(dataloader)
    ssim_avg = ssim_total / len(dataloader)
    print(f"Average PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}")
 
def check_data_range(dataloader):
    for lr_imgs, _ in dataloader:
        print("Min pixel value:", lr_imgs.min().item())
        print("Max pixel value:", lr_imgs.max().item())
        break

# Main function to train and save the model




