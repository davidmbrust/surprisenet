import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


class VideoFrameDataset(Dataset):
    def __init__(self, video_files, frame_rate, start_time, end_time, transform=None):
        self.video_files = video_files
        self.frame_rate = frame_rate
        self.start_time = start_time
        self.end_time = end_time
        self.transform = transform
        self.data, self.labels = self.load_frames()

    def load_frames(self):
        videos = [extract_specific_time_frames(vf, self.frame_rate, self.start_time, self.end_time)
                  for vf in self.video_files]
        min_length = min(len(v) for v in videos)  # Find the minimum sequence length
        trimmed_videos = [torch.stack([self.transform(frame) for frame in video[:min_length]])
                          for video in videos]
        labels = [torch.tensor(idx) for idx in range(len(videos))]
        return trimmed_videos, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def extract_specific_time_frames(video_path, frame_rate, start_time, end_time):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_interval = max(int(fps / frame_rate), 1)
    frames = []

    for frame_num in range(start_frame, end_frame + 1, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()
    return frames


def collate_fn(data):
    sequences, labels = zip(*data)
    sequence_lengths = torch.tensor([s.size(0) for s in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, labels, sequence_lengths


# Video files
video_files = ['./Videos/City_Video.mp4', './Videos/Forest_Video.mp4', './Videos/Ocean_Video.mp4', './Videos/Stock_Video.mp4']

# Load mean and std
mean_std = torch.load('./preprocessed_data/encoder_mean_std.pth')
mean, std = mean_std['mean'], mean_std['std']

# Transforms
normalized_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Getting Loader
video_dataset = VideoFrameDataset(video_files, frame_rate=24, start_time=10, end_time=20, transform=normalized_transform)
video_loader = DataLoader(video_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
