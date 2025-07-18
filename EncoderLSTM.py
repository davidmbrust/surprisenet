import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms
from VideoDataLSTM import video_loader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    
class EncoderLSTM(nn.Module):
    def __init__(self, encoder, lstm_hidden_size, num_lstm_layers, output_size, image_size, dropout=0.5):
        super(EncoderLSTM, self).__init__()

        # Identify the output channels of the last Conv layer
        conv_layers = [layer for layer in encoder if isinstance(layer, nn.Conv2d)]
        conv_out_channels = conv_layers[-1].out_channels if conv_layers else 128  # Default to 128 if no Conv layer
        
        # Calculate the output size of the encoder
        with torch.no_grad():
            example_input = torch.zeros(1, 3, *image_size)
            example_output = encoder(example_input)
            encoder_output_size = example_output.view(-1).size(0)

        self.encoder = encoder
        self.lstm = nn.LSTM(input_size=encoder_output_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                             dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        batch_size, seq_length, C, H, W = x.size()
        encoded_features = []
        for t in range(seq_length):
            frame = x[:, t, :, :, :]
            with torch.no_grad():
                encoded_frame = self.encoder(frame)
            encoded_features.append(encoded_frame)
        lstm_input = torch.stack(encoded_features, dim=1)
        lstm_input = lstm_input.view(batch_size, seq_length, -1)
        lstm_output, hidden_state = self.lstm(lstm_input, hidden_state)
        last_output = lstm_output[:, -1, :]
        output = self.fc(last_output)
        probabilities = F.softmax(output, dim=1)
        return probabilities, hidden_state

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

# Initialize with Xavier initialization
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

class ConvLSTMClassifier(nn.Module):
    def __init__(self, encoder, lstm_hidden_size: int, lstm_num_layers: int, num_classes: int):
        super(ConvLSTMClassifier, self).__init__()
        # Get the output size of the encoder
        with torch.no_grad():
            example_input = torch.zeros(1, 3, 64, 64)
            encoder_output = encoder(example_input)
            encoder_output_size = encoder_output.view(-1).size(0)

        self.encoder = encoder
        self.lstm = nn.LSTM(input_size=encoder_output_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.apply(init_weights) # Xavier initialization

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor] = None):
        batch_size, seq_length, C, H, W = x.size()
        encoded_features = []
        for t in range(seq_length):
            frame = x[:, t, :, :, :]
            with torch.no_grad():
                encoded_frame = self.encoder(frame)
            encoded_features.append(encoded_frame)
        lstm_input = torch.stack(encoded_features, dim=1)
        lstm_input = lstm_input.view(batch_size, seq_length, -1)
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden_state = self.lstm(packed_input, hidden_state)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_output = lstm_output[torch.arange(batch_size), lengths - 1, :]
        output = self.fc(last_output)
        probabilities = F.softmax(output, dim=1)
        return probabilities, hidden_state
    

def label_to_index(label, label_mapping):
    return label_mapping[label]

def train_lstm_classifier(model, dataloader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5,
        threshold=0.0001, threshold_mode='rel'
    )
    total = len(dataloader.dataset)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0

        for frames, labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct_predictions / total
        scheduler.step(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f'Epoch {epoch + 1} || Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | LR: {current_lr:.6f} | Time: {current_time}')
        torch.save(model.state_dict(), f'./LSTM_models/v2/LSTM_classifier_time_{current_time}_epoch_{epoch+1}_lr_{current_lr:.6f}_loss_{epoch_loss:.4f}_accuracy_{epoch_accuracy:.4f}.pth')

def train_conv_lstm_classifier(model, dataloader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5,
        threshold=0.0001, threshold_mode='rel'
    )
    total = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0

        for frames, labels, lengths in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(frames, lengths)
            loss = criterion(outputs, torch.stack(labels))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == torch.stack(labels)).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct_predictions / total
        scheduler.step(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f'Epoch {epoch + 1} || Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | LR: {current_lr:.6f} | Time: {current_time}')
        torch.save(model.state_dict(), f'./LSTM_models/v3/conv_lstm_model_time_{current_time}_epoch_{epoch + 1}_loss_{epoch_loss:.4f}_accuracy_{epoch_accuracy:.4f}.pth')


# Getting encoder layer
autoencoder = ConvAutoencoder()
autoencoder.load_state_dict(torch.load('./encoder_models/v9/AutoEncoder_model_time_2024-05-04 20:30:23_epoch_100_lr_0.007000_loss_0.5106.pth'))
autoencoder.eval()
encoder = autoencoder.encoder

# Freezing Layers
for param in encoder.parameters():
    param.requires_grad = False

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

# Choice of video files with labels
video_label_map = {"City": 0, "Forest": 1, "Ocean": 2, "Stock": 3}
video_files = ['./Videos/City_Video.mp4', './Videos/Forest_Video.mp4', 
               './Videos/Ocean_Video.mp4', './Videos/Stock_Video.mp4']
video_labels = ["City", "Forest", "Ocean", "Stock"]

# Set parameters
epochs = 100
learning_rate = 0.007
batch_size = 32

# Initializing model
#model = EncoderLSTM(encoder=encoder, lstm_hidden_size=64, num_lstm_layers=2, output_size=4, image_size=(64, 64))
conv_lstm_classifier = ConvLSTMClassifier(encoder, lstm_hidden_size=64, lstm_num_layers=2, num_classes=4)

# Training
#train_lstm_classifier(model, video_loader, epochs=epochs, learning_rate=learning_rate)
train_conv_lstm_classifier(conv_lstm_classifier, video_loader, epochs=epochs, learning_rate=learning_rate)
''''
for video, label in video_dataset:
    print(f"Video Label: {label}, Number of Frames: {video.size(0)}")

with torch.no_grad():
    example_input = torch.zeros(1, 3, 64, 64)
    encoder_output = encoder(example_input)
    encoder_output_size = encoder_output.view(-1).size(0)
    print(f"Encoder output size: {encoder_output_size}")
'''
