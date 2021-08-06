import argparse
import pandas as pd
import os, subprocess
from pytube import YouTube
from tqdm import tqdm


def download(data_path):
    print(f'Download data from {data_path} file')
    data = pd.read_csv(data_path)
    for id, song in tqdm(data.iterrows(), total=len(data)):
        dir_path = f"audio/{song['class']}/"
        video = YouTube(song['url'])
        audio = video.streams.filter(only_audio=True)
       
        # download audio file .mp4
        audio[0].download(dir_path, filename=str(id) + '.mp4')
        
        # convert to .wav file
        subprocess.run([
            'ffmpeg', '-i', 
            os.path.join(dir_path, str(id) + '.mp4'),
            os.path.join(dir_path, str(id) + '.wav'),
            '-hide_banner', '-loglevel', 'error'
        ])
        os.remove(os.path.join(dir_path, str(id) + '.mp4'))
        
        # segment into 15 minutes each
        subprocess.run([
            'ffmpeg', '-i', os.path.join(dir_path, str(id) + '.wav'), '-f', 
            'segment', '-segment_time', '900', '-c', 'copy', os.path.join(dir_path, str(id) + '-%04d' + '.wav'), 
            '-hide_banner', '-loglevel', 'error'
        ])
        os.remove(os.path.join(dir_path, str(id) + '.wav'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', type=str, default='data.csv')
    args = parser.parse_args()

    # download youtube audio
    download(args.data)

