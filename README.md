# Torch in Action

This repository contains the code for the Torch in Action book. 

[Chapter 1](01-Meeting-Torch): Meeting Torch
 * [facedetect](01-Meeting-Torch/facedetect): toy face detection dataset (directory with only four samples);
 * [train.lua](01-Meeting-Torch/train.lua): code for listing 1.1, 1.2 and 1.3;

[Chapter 2](02-Preparing-a-dataset): Preparing a dataset
 * [mnist](02-Preparing-a-dataset/mnist): MNIST dataset in binary format as downloaded from [yann.lecun.com](http://yann.lecun.com/exdb/mnist/);
 * [createdataset.lua](02-Preparing-a-dataset/createdataset.lua): code for serializing the MNIST dataset into `.t7` files and generating samples (section 2.3);
 * [dataloader.lua](02-Preparing-a-dataset/dataloader.lua): code for listing 2.1, 2.2, 2.3 and 2.5. Defines the `DataLoader` and `TensorLoader` classes);
 * [iteratedataset.lua](02-Preparing-a-dataset/iteratedataset.lua): code for listing 2.5. This script tests the `dataloader.lua` file by iterating through it. Only works if `createdataset.lua` was executed before hand;
 * [getmnistsample.lua](02-Preparing-a-dataset/getmnistsample.lua): bonus code for generating MNIST samples consolidated as a single image (used to generate figure 2.1);
