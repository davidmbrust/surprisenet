import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from VideoData import video_loader
import os
from PIL import Image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder: Reduce dimensionality more quickly
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # Output: [8, 64, 64]
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # Output: [16, 32, 32]
            nn.ReLU()
        )
        # Decoder: Simplified to match the encoder's output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # Output: [8, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 2, stride=2),  # Output: [3, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class ConvAutoencoder2(nn.Module):
    def __init__(self):
        super(ConvAutoencoder2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # Output: [32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: [64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Output: [128, 16, 16]
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: [64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: [32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # Output: [3, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Loading the model
model = ConvAutoencoder()
model.load_state_dict(torch.load('./encoder_models/v9/AutoEncoder_model_time_2024-05-04 20:30:23_epoch_100_lr_0.007000_loss_0.5106.pth'))  # Ensure the path to model is correct
model.eval()

# Load mean and std
mean_std = torch.load('./preprocessed_data/encoder_mean_std.pth')
mean, std = mean_std['mean'], mean_std['std']

# get transforms
normalized_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Resize frames
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std) # Using Computed mean and std
])


# Visualize original and reconstructed images
dataiter = iter(video_loader)
images = next(dataiter)

# Get reconstructed
with torch.no_grad():
    reconstructed = model(images)
    encoder = model.encoder(images)

def imshow(img_tensor, title="Image", show=True, filename=None):
    img = img_tensor.cpu().detach().numpy()  # Convert to numpy
    img = img.transpose((1, 2, 0))  # Convert to HWC format
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

    if show:
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    if filename:
        img_pil = Image.fromarray((img * 255).astype('uint8'))
        img_pil.save(filename)
    
    plt.show()

def show_encoded_features_grid(encoded, num_features=6, grid_shape=(2, 3)):
    encoded = encoded.cpu().detach().numpy()
    if len(encoded.shape) == 3:  # For a single image (channels, height, width)
        channels, height, width = encoded.shape
        encoded = encoded.reshape((1, channels, height, width))
    batch_size, channels, height, width = encoded.shape

    num_features = min(num_features, channels)
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(12, 6))
    axes = axes.flatten()

    for i in range(min(num_features, grid_shape[0] * grid_shape[1])):
        ax = axes[i]
        ax.imshow(encoded[0, i, :, :], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Feature Map {i+1}')

    # Turn off any remaining empty subplots
    for j in range(i + 1, grid_shape[0] * grid_shape[1]):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def show_encoded_features(encoded, num_features=6):
    """
    Visualize the encoded feature maps as grayscale images.
    """
    encoded = encoded.cpu().detach().numpy()
    if len(encoded.shape) == 3:  # For a single image (channels, height, width)
        channels, height, width = encoded.shape
        encoded = encoded.reshape((1, channels, height, width))
    batch_size, channels, height, width = encoded.shape

    for i in range(min(num_features, channels)):
        plt.imshow(encoded[0, i, :, :], cmap='gray')
        plt.title(f'Encoded Feature Map {i+1}')
        plt.axis('off')
        plt.show()

def show_feature_maps_grid(feature_maps, num_maps=6, title="Feature Maps"):
    """
    Visualize feature maps in a grid layout.
    """
    num_maps = min(num_maps, feature_maps.size(1))
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    
    # Ensure axes is iterable even when there's only one subplot
    if num_maps == 1:
        axes = [axes]

    for i in range(num_maps):
        axes[i].imshow(feature_maps[0, i, :, :].cpu().detach().numpy(), cmap='gray')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

def show_encoded_features_average(encoded):
    """
    Display the average of all encoded feature maps as a grayscale image.
    """
    average_map = torch.mean(encoded, dim=1).values
    print(average_map)
    average_map = average_map.numpy()
    plt.imshow(average_map, cmap='gray')
    plt.title('Average of Encoded Feature Maps')
    plt.axis('off')
    plt.show()

def show_encoded_features_max(encoded):
    """
    Display the maximum intensity projection of all encoded feature maps as a grayscale image.
    """
    max_map = torch.max(encoded, dim=1).values
    max_map = max_map.squeeze().cpu().detach().numpy()
    plt.imshow(max_map, cmap='gray')
    plt.title('Maximum Intensity Projection of Encoded Feature Maps')
    plt.axis('off')
    plt.show()

i = 0
for i in range(0,31):
    imshow(images[i], "Original Image", filename=f"./image_reconstructions/original_{i}.png")
    imshow(reconstructed[i], "Reconstructed Image", filename=f"./image_reconstructions/reconstructed_{i}.png")
i = 0
#imshow(images[i], "Original Image", filename=f"./image_reconstructions/original_{i}.png")
#imshow(reconstructed[i], "Reconstructed Image", filename=f"./image_reconstructions/reconstructed_{i}.png")
#show_encoded_features_grid(encoder[i])

#for i in range(0,31):
#    show_encoded_features_grid(encoder[i])
