import argparse
from genericpath import exists
import os, subprocess
import shutil
from pydub import AudioSegment


def source_separate(audio_dir):
    command = ["python", "-m", "demucs.separate", "-n", "demucs_quantized"]
    for cls in os.listdir(audio_dir):
        cls_dir = os.path.join(audio_dir, cls)
        os.makedirs(f'music/{cls}', exist_ok=True)
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.wav'):
                continue

            # separate audio
            cmd = command.copy() + [os.path.join(cls_dir, fname)]
            subprocess.run(cmd)

            # overlay music tracks
            name = fname.split('.')[0]
            bass = AudioSegment.from_file(f"separated/demucs_quantized/{name}/bass.wav", format="wav")
            drums = AudioSegment.from_file(f"separated/demucs_quantized/{name}/drums.wav", format="wav")
            other = AudioSegment.from_file(f"separated/demucs_quantized/{name}/other.wav", format="wav")
            music = bass.overlay(drums)
            music = music.overlay(other)

            # save audio
            music.export(f"music/{cls}/{fname}", format="wav")

            if os.path.exists('separated'):
                shutil.rmtree('separated')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', action='store', type=str, default='audio')
    args = parser.parse_args()

    # source separation
    source_separate(args.audio_dir)

