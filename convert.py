import argparse
import numpy as np
import os, subprocess
from pydub import AudioSegment
import shutil
import soundfile as sf

import torch
from model.generator import Generator
from utils import Audio


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

audio = Audio()

gen = Generator().to(device)
checkpoint = torch.load('checkpoints/jazz-to-city-pop.pt')
gen.load_state_dict(checkpoint['gen'])


def separate_source(path):
    command = ["python", "-m", "demucs.separate", "-n", "demucs_quantized", path]
    subprocess.run(command)

    # overlay music tracks
    name = path.split('/')[-1].split('.')[0]
    bass = AudioSegment.from_file(f'separated/demucs_quantized/{name}/bass.wav', format='wav')
    drums = AudioSegment.from_file(f'separated/demucs_quantized/{name}/drums.wav', format='wav')
    other = AudioSegment.from_file(f'separated/demucs_quantized/{name}/other.wav', format='wav')
    music = bass.overlay(drums)
    music = music.overlay(other)

    # get vocal track
    vocal = AudioSegment.from_file(f'separated/demucs_quantized/{name}/vocals.wav', format='wav')
    
    # save audio temporally
    os.makedirs('tmp', exist_ok=True)
    music.export(f'tmp/music.wav', format='wav')
    vocal.export(f'tmp/vocal.wav', format='wav')

    if os.path.exists('separated'):
        shutil.rmtree('separated')


def inference(mel):
    tensor = torch.from_numpy(np.expand_dims(mel, [0, 1])).to(device)
    output = gen(tensor).data.cpu().numpy()
    return output[0][0]


def convert():
    # load mel-spectrogram of music
    mel = audio.audio_to_mel('tmp/music.wav')

    # inference chunks
    n = mel.shape[-1]
    length = 384
    pos = 0
    generated = np.zeros_like(mel)
    while pos < n:
        pos = min(pos, n - length)
        generated[:, pos: pos+length] = inference(mel[:, pos: pos+length])
        pos += length
    return generated


def save_mel(mel):
    res = audio.mel_to_audio(mel)
    sf.write('tmp/music-converted.wav', res, samplerate=22050)


def finalize(out_path):
    music = AudioSegment.from_file('tmp/music-converted.wav', format='wav')
    vocal = AudioSegment.from_file('tmp/vocal.wav', format='wav')

    track = music.overlay(vocal)
    track.export(out_path, format='wav')

    shutil.rmtree('tmp')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', type=str, required=True)
    parser.add_argument('--out_path', action='store', type=str, required=True)

    args = parser.parse_args()

    # 1. extract music from 
    separate_source(args.source)

    # 2. convert audio waveform to mel-spectrogram
    converted_mel = convert()

    # 3. save converted mel
    save_mel(converted_mel)

    # 4. combine music and vocal
    finalize(args.out_path)
