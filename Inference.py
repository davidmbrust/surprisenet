import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, List
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from VideoDataLSTM import video_dataset, collate_fn

# Model Definition
class ConvLSTMClassifier(nn.Module):
    def __init__(self, encoder, lstm_hidden_size, lstm_num_layers, num_classes):
        super(ConvLSTMClassifier, self).__init__()
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
        self.apply(init_weights)

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


# Load the encoder
encoder = torch.load("encoder.pth")

# Create the ConvLSTMClassifier
model = ConvLSTMClassifier(encoder, lstm_hidden_size=64, lstm_num_layers=2, num_classes=6)
model.load_state_dict(torch.load("conv_lstm_model.pth"))
model.eval()

# Video Labels
video_labels = ["City", "Forest", "Ocean", "Stock"]

# Creating Combinations
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


# Create combined videos dataset
combined_videos, combined_labels = create_all_combinations(video_dataset)

# Create the dataloader
combined_video_loader = DataLoader(combined_videos, batch_size=1, shuffle=False, collate_fn=collate_fn)


# Run Inference
def run_inference(model, frames):
    hidden = model.init_hidden()
    outputs = []
    states = []
    for frame in frames:
        frame_tensor = frame.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        output, hidden = model(frame_tensor, lengths=torch.tensor([1]), hidden_state=hidden)
        outputs.append(output)
        states.append(hidden)
    return outputs, states


outputs_combined, states_combined = zip(*[run_inference(model, frames) for frames, _ in combined_videos])


# Plot Functions
def plot_probabilities(outputs, labels, titles):
    for output, label, title in zip(outputs, labels, titles):
        probs = [F.softmax(o, dim=1).detach().numpy() for o in output]
        avg_probs = np.mean(probs, axis=0)
        plt.figure(figsize=(10, 5))
        for i, prob in enumerate(zip(*probs)):
            plt.plot(prob, label=f'Class {video_labels[i]}')
        plt.plot(avg_probs, linestyle='--', color='black', label='Average')
        plt.xlabel('Frame Number')
        plt.ylabel('Probability')
        plt.title(f"{title} ({label})")
        plt.legend()
        plt.show()


def plot_hidden_states(states, labels, titles):
    for state, label, title in zip(states, labels, titles):
        hidden, cell = zip(*state)
        hidden_states = [h[0].detach().numpy() for h in hidden]
        avg_hidden_states = np.mean(hidden_states, axis=0)
        plt.figure(figsize=(10, 5))
        for i, hidden_state in enumerate(zip(*hidden_states)):
            plt.plot(hidden_state, label=f'Hidden Unit {i+1}')
        plt.plot(avg_hidden_states, linestyle='--', color='black', label='Average')
        plt.xlabel('Frame Number')
        plt.ylabel('Activation')
        plt.title(f"{title} ({label})")
        plt.legend()
        plt.show()


# Plot probabilities
plot_probabilities(outputs_combined, combined_labels, [f'Combined Video {i+1} Probabilities' for i in range(len(outputs_combined))])

# Plot hidden states
plot_hidden_states(states_combined, combined_labels, [f'Combined Video {i+1} Hidden States' for i in range(len(states_combined))])
