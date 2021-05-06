# Single Image Super-Resolution with SRGAN (Incomplete)

## Introduction

This project is a reimplementation of SRGAN forked from [this repo](https://krasserm.github.io/2019/09/04/super-resolution/) for super resolution. The only difference is that we used Keras model subclassing to implement SRGAN model instead of Functional API.

## Getting started 

Examples in this section require following pre-trained weights for running (see also example notebooks):  

### Pre-trained weights

- [weights-srgan.tar.gz](https://martin-krasser.de/sisr/weights-srgan.tar.gz) 
    - SRGAN as described in the SRGAN paper: 1.55M parameters, trained with VGG54 content loss.
    
After download, extract them in the root folder of the project with

    tar xvfz weights-<...>.tar.gz


### SRGAN

```python
from model.srgan import generator

model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

lr = load_image('demo/0869x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-srgan](docs/images/result-srgan.png)

## DIV2K dataset

For training and validation on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) images, applications should use the 
provided `DIV2K` data loader. It automatically downloads DIV2K images to `.div2k` directory and converts them to a 
different format for faster loading.

### Training dataset

```python
from data import DIV2K

train_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train')      # Training dataset are images 001 - 800
                     
# Create a tf.data.Dataset          
train_ds = train_loader.dataset(batch_size=16,         # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

# Iterate over LR/HR image pairs                                
for lr, hr in train_ds:
    # .... 
```

Crop size in HR images is 96x96. 

### Validation dataset

```python
from data import DIV2K

valid_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid')      # Validation dataset are images 801 - 900
                     
# Create a tf.data.Dataset          
valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch
                                
# Iterate over LR/HR image pairs                                
for lr, hr in valid_ds:
    # ....                                 
```

## Training 

The following training examples use the [training and validation datasets](#div2k-dataset) described earlier. The high-level 
training API is designed around *steps* (= minibatch updates) rather than *epochs* to better match the descriptions in the 
papers.

## SRGAN

### Generator pre-training

```python
from model.srgan import generator
from train import SrganGeneratorTrainer

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')

# Pre-train the generator with 1,000,000 steps (100,000 works fine too). 
pre_trainer.train(train_ds, valid_ds.take(10), steps=1000000, evaluate_every=1000)

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')
```

### Generator fine-tuning (GAN)

```python
from model.srgan import generator, discriminator
from train import SrganTrainer

# Create a new generator and init it with pre-trained weights.
gan_generator = generator()
gan_generator.load_weights('weights/srgan/pre_generator.h5')

# Create a training context for the GAN (generator + discriminator).
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

# Train the GAN with 200,000 steps.
gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/gan_generator.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator.h5')
```
