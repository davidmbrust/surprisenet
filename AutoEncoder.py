import torch
import torch.nn as nn
import torch.optim as optim
from VideoData import video_loader
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import torch
import torch.nn as nn
from AutoEncoder import ConvAutoencoder


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
    

def train_autoencoder(model, dataloader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total = len(dataloader.dataset)

    # Getting learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.80, patience=5,
        threshold=0.0001, threshold_mode='rel')

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for frames in dataloader:
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, frames)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frames.size(0)
        
        epoch_loss = running_loss / total
        
        # Adjust learning rate
        scheduler.step(epoch_loss)

        # Get leraning rate adn time
        current_lr = scheduler.get_last_lr()[0]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Output epoch
        print(f'Epoch {epoch+1} || Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Time: {current_time}')

        # Save model at epoch
        torch.save(model.state_dict(), f'./encoder_models/v9/AutoEncoder_model_time_{current_time}_epoch_{epoch+1}_lr_{current_lr:.6f}_loss_{epoch_loss:.4f}.pth')


# Set parameters
batch_size = 32
epochs = 100
learning_rate = 0.007

# Initialize the model
autoencoder = ConvAutoencoder()

# Train the model
train_autoencoder(autoencoder, video_loader, epochs, learning_rate)