import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VideoDataLSTM import video_dataset, collate_fn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

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
    
    def init_hidden(self, batch_size=1):
        hidden_state = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                        torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))
        return hidden_state


# Prepare the combinations
def create_all_combinations(video_dataset):
    min_length = min([len(v) for v in video_dataset.data])
    half_idx = min_length // 2

    combinations = []
    labels = []
    label_counter = 0

    for i in range(4):
        for j in range(4):
            # Combine the first half of video i with the second half of video j
            combined_data = torch.cat((video_dataset.data[i][:half_idx], video_dataset.data[j][half_idx:]), dim=0)
            combinations.append((combined_data, torch.tensor(label_counter)))
            labels.append(f"{video_labels[i]}-{video_labels[j]}")
            label_counter += 1

    for idx, (data, label) in enumerate(zip(video_dataset.data, video_dataset.labels)):
        combinations.append((data, label))
        labels.append(video_labels[idx])

    return combinations, labels

def collate_fn(data):
    sequences, labels = zip(*data)
    sequence_lengths = torch.tensor([s.size(0) for s in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, labels, sequence_lengths

def run_inference(model, frames):
    model.eval()
    hidden = model.init_hidden()
    outputs = []
    states = []
    for frame in frames:
        frame_tensor = frame.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        output, hidden = model(frame_tensor, lengths=torch.tensor([1]), hidden_state=hidden)
        outputs.append(output)
        states.append(hidden)
    return outputs, states


# Video Labels
video_labels = ["City", "Forest", "Ocean", "Stock"]

# Create combined videos dataset
combined_videos, combined_labels = create_all_combinations(video_dataset)


# Create the dataloader
combined_video_loader = DataLoader(combined_videos, batch_size=1, shuffle=False, collate_fn=collate_fn)


# Getting encoder layer
autoencoder = ConvAutoencoder()
autoencoder.load_state_dict(torch.load('./encoder_models/v9/AutoEncoder_model_time_2024-05-04 20:30:23_epoch_100_lr_0.007000_loss_0.5106.pth'))
autoencoder.eval()
encoder = autoencoder.encoder

# Get LSTM Model
model = ConvLSTMClassifier(encoder, lstm_hidden_size=64, lstm_num_layers=2, num_classes=4)
model.load_state_dict(torch.load("./LSTM_models/v4/conv_lstm_model_time_2024-05-05 12:43:07_epoch_25_loss_0.7438_accuracy_1.0000.pth"))
model.eval()


# Run inference on all videos
outputs_combined, states_combined = zip(*[run_inference(model, frames) for frames, _ in combined_videos])

# Save data to file
torch.save({
    'outputs_combined': outputs_combined,
    'states_combined': states_combined,
    'combined_labels': combined_labels
}, './preprocessed_data/video_inference_data.pth')