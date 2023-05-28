import random
import torch
import torchaudio
import os
from music.nn.net import AlexNet
from music.nn.GTZANDataset import GTZANDataset

# Set the path for the annotations file
PATH_ANNOTATIONS_FILE = os.path.abspath("./music/nn/features_30_sec_final.csv")
# Set the path for the audio files
PATH_AUDIO_DIR = "./music/static/GTZAN/genres_original"
ANNOTATIONS_FILE = PATH_ANNOTATIONS_FILE
AUDIO_DIR = PATH_AUDIO_DIR
# Sample rate
SAMPLE_RATE = 22050
# Number of samples (5 seconds)
NUM_SAMPLES = 22050 * 5
# Initialize the model
cnn = AlexNet()
# Load the model
path_abs = os.path.abspath("./music/nn/best_model_okk.pth")
state_dict = torch.load(path_abs, map_location='cpu')
cnn.load_state_dict(state_dict)

# This is not used
mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=128,
    log_mels=True
)

# Convert to Mel spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# Data processing
gtzan = GTZANDataset(annotations_file=ANNOTATIONS_FILE, audio_dir=AUDIO_DIR,
                     transformation=mfcc, target_sample_rate=SAMPLE_RATE,
                     num_samples=NUM_SAMPLES, device="cpu")
# Label list
class_mapping = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

cnn.eval()

# This corresponds to the index.html page, i.e., the initial version
def get_music():
    global class_mapping
    # Get a random music index
    initial = random.sample(range(0, 1000), 1)[0]
    print(f'initial:{initial}')
    # Get the music label index
    music_init_index = gtzan[initial][1]
    # Get the music URL
    music_init_url = gtzan[initial][2]
    print(f'music_init:{music_init_url}')
    return class_mapping[music_init_index], music_init_url, music_init_index

# project.html final version: Preliminary encapsulation of music data
def get_music_list(initial):
    music_list = []
    for i in range(5):
        with torch.no_grad():
            # Dictionary to store the parameters
            per = {}
            # Get the genre name
            name = class_mapping[gtzan[initial[i]][1]]
            # Get the music URL
            init_url = gtzan[initial[i]][2]
            mp3 = init_url.replace("/music", '')
            # Music genre name
            per["name"] = name
            # Music name
            per["artist"] = init_url.rsplit("/", 1)[1]
            # Music URL
            per["mp3Url"] = mp3
            # Music ID, mainly used for frontend templates
            per["id"] = i
            # Music genre index
            per["style"] = int(gtzan[initial[i]][1])
            # Get a random background image
            per["picUrl"] = f'./static/background/background_{random.randint(0, 7)}.jpg'
            music_list.append(per)
    return music_list


# project.html: Get information for 5 randomly selected songs
def net_music_list():
    global class_mapping
    initial = random.sample(range(0, 1000), 5)
    return get_music_list(initial)

# project.html: Get information for 5 randomly selected songs of the same genre
def net_recommend_genre():
    global class_mapping
    # Randomly select a number from 0 to 9 as the genre index
    index = random.sample(range(0, 10), 1)[0]
    # Randomly select a number from 0 to 999 within the genre range
    initial = random.sample(range(index * 100, index * 100 + 99), 5)
    return get_music_list(initial)


# Recommend songs based on favorite songs
def net_predict_music(music_index):
    global class_mapping
    # Randomly select 20 indices
    ran = random.sample(range(0, 1000), 20)
    music_list = []
    # Pass these 20 songs through the neural network model
    # Sort the output values and select the top 5 values
    # It is recommended to modify AlexNet and add a SoftMax layer for better results
    for i in range(20):
        # Omitted code below
        with torch.no_grad():
            per = {}
            predictions = cnn(gtzan[ran[i]][0].unsqueeze_(0))
            predicted_item = predictions[0][music_index].item()
            name = class_mapping[gtzan[ran[i]][1]]
            per["value"] = predicted_item
            url = gtzan[ran[i]][2]
            mp3 = url.replace("/music", '')
            per["name"] = name
            per["artist"] = url.rsplit("/", 1)[1]
            per["mp3Url"] = mp3
            per["id"] = i
            per["style"] = int(gtzan[ran[i]][1])
            per["picUrl"] = f'./static/background/background_{random.randint(0, 7)}.jpg'
            music_list.append(per)
    # Sort the output values
    sort_music = sorted(music_list, key=lambda x: x.get("value"), reverse=True)
    # Select the top 5 values
    return sort_music[0:5]

# This corresponds to the index.html page. The logic is similar, recommending songs based on the selected song
def get_predict_music(music_init_index):
    # global music_init_index
    global class_mapping
    max_music_value = - float("inf")
    max_music_index = None
    max_music_url = None
    real_label_index = None
    # Randomly select 15 songs
    ran = random.sample(range(0, 1000), 15)
    content = {}
    for i in range(15):
        with torch.no_grad():
            predictions = cnn(gtzan[ran[i]][0].unsqueeze_(0))
            predicted_item = predictions[0][music_init_index].item()
            # Record the information of the song with the maximum output value
            if max_music_value < predicted_item:
                max_music_value = predicted_item
                max_music_index = i
                real_label_index = gtzan[ran[i]][1]
                max_music_url = gtzan[ran[i]][2]
            content[gtzan[ran[i]][2]] = predicted_item
    return max_music_url, class_mapping[music_init_index], class_mapping[real_label_index]



if __name__ == "__main__":
    list_1 = [{"id":1},{"id":0},{"id":5}]
    r = sorted(list_1, key=lambda x: x.get("id"))
    print(r)
    # print(net_music_list())
