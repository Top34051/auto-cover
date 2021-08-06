import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split

from utils import Audio


def load_wav_audio(audio_dir):
    audio = Audio()
    print(f'Download audio data from "{audio_dir}" folder')
    data = {}
    for cls in os.listdir(audio_dir):
        cls_dir = os.path.join(audio_dir, cls)

        data[cls] = []
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.wav'):
                continue

            print('->', os.path.join(cls_dir, fname))

            # convert audio to mel
            mel = audio.audio_to_mel(os.path.join(cls_dir, fname))
            print('\tconvert to mel: done!')

            # get samples from mel-spectrogram of width 256
            k = mel.shape[1] // 30
            mels = audio.mel_sample(mel, width=256, k=k)
            print('\tsample mels: done!')

            # append samples
            if mels is not None:
                data[cls].append(mels)
                
        data[cls] = np.concatenate(data[cls], axis=0)
    return data

def create_dataset(data):
    os.makedirs('./data/train', exist_ok=True)
    os.makedirs('./data/test', exist_ok=True)
    for cls in data:
        train, test = train_test_split(data[cls], test_size=0.1, random_state=101)
        np.save(f'./data/train/{cls}.npy', train)
        np.save(f'./data/test/{cls}.npy', test)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', action='store', type=str, default='data/music')
    args = parser.parse_args()

    # 1. load wav audio
    data = load_wav_audio(args.audio_dir)

    # 2. create dataset
    create_dataset(data)