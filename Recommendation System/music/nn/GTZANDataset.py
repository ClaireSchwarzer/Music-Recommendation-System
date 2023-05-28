import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
import matplotlib.pyplot as plt

# Dataset class for processing the dataset, converting audio data to tensors, and maintaining consistent input states for each audio data
class GTZANDataset(Dataset):

   def __init__(self,
             annotations_file,
             audio_dir,
             transformation,
             target_sample_rate,
             num_samples,
             device):
    # Read the label file
    self.annotations = pd.read_csv(annotations_file)
    # Read the audio directory
    self.audio_dir = audio_dir
    # Set the device
    self.device = device
    # Load the Mel-frequency spectrogram data into the device
    self.transformation = transformation.to(self.device)
    # Set the target sample rate
    self.target_sample_rate = target_sample_rate
    # Set the number of samples
    self.num_samples = num_samples

# Return the number of audio files
def __len__(self):
    return len(self.annotations)

# Access the audio data, label, and path as an array
def __getitem__(self, index):
    # Get the audio sample path
    audio_sample_path = self._get_audio_sample_path(index)
    # Process the path to ensure correct format
    path_address = audio_sample_path.replace("\\", '/')
    # Get the label
    label = self._get_audio_sample_label(index)
    # Load the signal and sample rate
    signal, sr = torchaudio.load(audio_sample_path)
    signal = signal.to(self.device)
    # Adjust the sample rate
    signal = self._resample_if_necessary(signal, sr)
    # Convert stereo to mono
    signal = self._mix_down_if_necessary(signal)
    # Adjust the sample length
    signal = self._cut_if_necessary(signal)
    signal = self._right_pad_if_necessary(signal)
    # Transform to Mel-frequency spectrogram
    signal = self.transformation(signal)
    return signal, label, path_address

# Check if signal needs to be cropped: If the sample length > the set length, then crop it
def _cut_if_necessary(self, signal):
    # print('_cut_if_necessary')
    if signal.shape[1] > self.num_samples:
        signal = signal[:, :self.num_samples]
    return signal

# Check if signal needs to be padded: If the sample length < the set length, then pad it with zeros on the right
def _right_pad_if_necessary(self, signal):
    length_signal = signal.shape[1]
    # print('_right_pad_if_necessary')
    if length_signal < self.num_samples:
        num_missing_samples = self.num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        # last_dim_padding.to(self.device)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

# Resample the signal if necessary
def _resample_if_necessary(self, signal, sr):
    # print('_resample_if_necessary')
    # Only resample if the actual sample rate is different from the set sample rate
    if sr != self.target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
        signal = resampler(signal)
        # signal = torchaudio.functional.resample(signal, sr, self.target_sample_rate)

    return signal

# Convert stereo signal to mono
def _mix_down_if_necessary(self, signal):
    # print('_mix_down_if_necessary')
    # Take the mean of the channels to convert stereo to mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

# Get the audio sample path by joining the necessary parts
def _get_audio_sample_path(self, index):
    # print('_get_audio_sample_path')
    fold = f"{self.annotations.iloc[index, -2]}"
    path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
        index, 1])
    return path

# Extract the label from the CSV file
def _get_audio_sample_label(self, index):
    # print('_get_audio_sample_label')
    return self.annotations.iloc[index, -1]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "features_30_sec_final.csv"
    AUDIO_DIR = "F:\datasets\GTZAN\genres_original"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 * 5  # -> 1 second of audio
    plot = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        log_mels=True
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # objects inside transforms module are callable!
    # ms = mel_spectrogram(signal)

    gtzan = GTZANDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    print(f"There are {len(gtzan)} samples in the dataset")

    if plot:
        signal, label, path = gtzan[666]
        print(f'path:{path}')
        signal = signal.cpu()
        print(signal.shape)

        plt.figure(figsize=(16, 8), facecolor="white")
        plt.imshow(signal[0, :, :], origin='lower')
        plt.autoscale(False)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.axis('auto')
        plt.show()
