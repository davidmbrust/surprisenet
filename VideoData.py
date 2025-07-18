import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def extract_frames(video_path, frame_rate):
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    every_x_frame = max(1, round(fps / frame_rate))
    
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        if i % every_x_frame == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB
    video.release()
    return frames

def extract_specific_time_frames(video_path, frame_rate, start_time, end_time):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the starting and ending frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Calculate the frame interval for the desired frame rate
    frame_interval = max(int(fps / frame_rate), 1)

    # Set the video position to the starting frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []

    while True:
        ret, frame = video.read()
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if not ret or current_frame > end_frame:
            break

        # Save the frame if it's at the correct interval
        if (current_frame - start_frame) % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()
    return frames

class VideoFrameDataset(Dataset):
    def __init__(self, video_files, frame_rate, transform=None):
        self.video_files = video_files
        self.frame_rate = frame_rate
        self.transform = transform
        self.frames = []
        self.load_frames()

    def load_frames(self):
        # Specify the time window you want to load frames from
        for video_file in self.video_files:
            self.frames.extend(extract_specific_time_frames(video_file, self.frame_rate, 10, 20))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

# Function for normalization parameters
def compute_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in loader:
        # Data shape is (batch_size, channels, height, width)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


# Set up transforms
not_normalized_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Resize frames
    transforms.ToTensor()
])

# Choice of video files
video_files = ['./Videos/City_Video.mp4', './Videos/Forest_Video.mp4', './Videos/Ocean_Video.mp4', './Videos/Stock_Video.mp4']

# Get non-normalized dataloader
video_dataset_not_normalized = VideoFrameDataset(video_files, frame_rate=5, transform=not_normalized_transform)
video_loader_not_normalized = DataLoader(video_dataset_not_normalized, batch_size=32, shuffle=True)

# Get normalization parameters
mean, std = compute_mean_std(video_loader_not_normalized)

# Save mean and std
torch.save({'mean': mean, 'std': std}, './preprocessed_data/encoder_mean_std.pth')

# Get new transforms
normalized_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Resize frames
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std) # Using Computed mean and std
])

# Get normalized dataloader
video_dataset = VideoFrameDataset(video_files, frame_rate=24, transform=normalized_transform)
video_loader = DataLoader(video_dataset, batch_size=32, shuffle=True)