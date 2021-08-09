# From Jazz To Japanese City Pop

This repository is aiming to convert jazz music into Japanese city pop style using Generative Adversarial Network architecture. The audio data used in this downloaded YouTube videos using PyTube library. 

Generator and discriminator are trained to do conversion from domain A to domain B and to distinguish between real domain B audio sample. I also train the network to smooth out between inferenced segments in the converted music. This is done by splitting and combining technique used while training. U-Net architecture is selected for the generator; while discriminator and embedding network are CNN models.
