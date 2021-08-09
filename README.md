# From Jazz To Japanese City Pop

This repository is aiming to convert jazz music into Japanese city pop style using Generative Adversarial Network architecture. The audio data used in this downloaded YouTube videos using PyTube library. 

Generator and discriminator are trained to do a conversion from domain A to domain B and to distinguish between real domain B audio samples. I also train the network to smooth out between inferences segments in the converted music. This is done by splitting and combining techniques used while training. U-Net architecture is selected for the generator; while discriminator and embedding network is CNN models.

**Disclaimer**: the work is still in progress. The quality of the conversion might not be as good.

## Style Transfering

To convert from Jazz to Japanese city pop, you will need a trained generator weight, which you can download from here: [link](https://drive.google.com/drive/folders/16nthg1wkeQV5b4hhtD6QVrSATjgRYrtt?usp=sharing). Place this checkpoint in `checkpoints/` folder and run
```bash
python convert.py \
  --source path_to_jazz.wav \
  --out_path results/jazz-to-city-pop/output.wav
```

## Training

First, you need a dataset. You can modify `data/data.csv` by specifying genre (class) and YouTube music Url. After that, run 
```bash
cd data
python download --data data.csv
python separate --audio_dir audio
```

These commands will download the audio files, place them in `data/audio/` folder, and perform source separation to remove a singing voice from the audio files.

Now, you can modify the training config in `config.json` and run 
```bash
python main.py --config_file config.json
```
to train the agents with our separated audio data.

## Results

August 9, 2021: The model is succeeded in converting the Jazz track to a new style while keeping the old content. We can see that it is adding drum rhythms as we usually found in Japanese city pop and the instrument sound has changed a bit. However, this architecture has a huge room to improve. The big concern is that the converted audio adopts little to no synth sound from city pop.

For your reference: please visit `examples/` folder. `city-pop-1.wav` has the same content as `jazz-1.wav` but is converted to Japanese city pop style.