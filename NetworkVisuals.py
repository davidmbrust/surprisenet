import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_convlstm_classifier(model):
    """
    Visualize the ConvLSTMClassifier model.
    """

    def draw_box(ax, pos, width, height, label, color='skyblue'):
        """
        Draws a box at the specified position with a given width, height, and label.
        """
        box = patches.FancyBboxPatch(
            (pos, 0), width, height, boxstyle="round,pad=0.02", color=color, alpha=0.5
        )
        ax.add_patch(box)
        ax.text(pos + width / 2, height / 2, label, ha="center", va="center")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    layers = []
    if isinstance(model.encoder, nn.Sequential):
        for layer in model.encoder:
            layers.append(layer)
    else:
        for layer in model.encoder.encoder:
            layers.append(layer)
    layers.append(model.lstm)
    layers.append(model.fc)

    pos = 0
    max_height = max([layer.out_channels if hasattr(layer, 'out_channels') else layer.hidden_size for layer in layers if hasattr(layer, 'out_channels') or hasattr(layer, 'hidden_size')])

    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            label = f"{layer.__class__.__name__}\n{layer.in_channels} -> {layer.out_channels}"
            height = layer.out_channels / max_height * 4
        elif isinstance(layer, nn.LSTM):
            label = f"{layer.__class__.__name__}\n{layer.input_size} -> {layer.hidden_size}"
            height = layer.hidden_size / max_height * 4
        elif isinstance(layer, nn.Linear):
            label = f"{layer.__class__.__name__}\n{layer.in_features} -> {layer.out_features}"
            height = layer.out_features / max_height * 4
        else:
            continue
        draw_box(ax, pos, 1, height, label)
        pos += 1.5

    ax.set_xlim(-1, pos)
    ax.set_ylim(-1, 5)

    plt.tight_layout()
    plt.show()


# Example usage
class ExampleEncoder(nn.Module):
    def __init__(self):
        super(ExampleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvLSTMClassifier(nn.Module):
    def __init__(self, encoder, lstm_hidden_size: int, lstm_num_layers: int, num_classes: int):
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor] = None):
        batch_size, seq_length, C, H, W = x.size()
        encoded_features = []
        for t in range(seq_length):
            frame = x[:, t, :, :, :]
            encoded_frame = self.encoder(frame)
            encoded_features.append(encoded_frame)
        lstm_input = torch.stack(encoded_features, dim=1)
        lstm_input = lstm_input.view(batch_size, seq_length, -1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden_state = self.lstm(packed_input, hidden_state)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        last_output = lstm_output[torch.arange(batch_size), lengths - 1, :]
        output = self.fc(last_output)
        probabilities = F.softmax(output, dim=1)
        return probabilities, hidden_state


def visualize_conv_autoencoder(model):
    """
    Visualize the ConvAutoencoder model.
    """

    def draw_box(ax, pos, width, height, label, color='skyblue'):
        """
        Draws a box at the specified position with a given width, height, and label.
        """
        box = patches.FancyBboxPatch(
            (pos, 0), width, height, boxstyle="round,pad=0.02", color=color, alpha=0.5
        )
        ax.add_patch(box)
        ax.text(pos + width / 2, height / 2, label, ha="center", va="center")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    layers = model.encoder + model.decoder
    max_channels = max(layer.out_channels for layer in layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d))

    pos = 0
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            label = f"{layer.__class__.__name__}\n{layer.in_channels} -> {layer.out_channels}"
            height = layer.out_channels / max_channels * 4
        elif isinstance(layer, nn.ReLU):
            label = f"{layer.__class__.__name__}"
            height = max_channels / max_channels * 4
        elif isinstance(layer, nn.Sigmoid):
            label = f"{layer.__class__.__name__}"
            height = max_channels / max_channels * 4
        else:
            continue
        draw_box(ax, pos, 1, height, label)
        pos += 1.5

    ax.set_xlim(-1, pos)
    ax.set_ylim(-1, 5)

    plt.tight_layout()
    plt.show()


# Define the ConvAutoencoder class
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


# Example usage
modelconv = ConvAutoencoder()

encoder = ExampleEncoder()
modellstm = ConvLSTMClassifier(encoder, 32, 2, 10)

visualize_convlstm_classifier(modellstm)
visualize_conv_autoencoder(modelconv)