# Torch in Action

This repository contains the code for the Torch in Action book. 

[Chapter 1](01-Meeting-Torch): Meeting Torch
 * [facedetect](01-Meeting-Torch/facedetect): toy face detection dataset (directory with only four samples);
 * [train.lua](01-Meeting-Torch/train.lua): example face detection training script (listings 1.1, 1.2 and 1.3);

[Chapter 2](02-Preparing-a-dataset): Preparing a dataset
 * [mnist](02-Preparing-a-dataset/mnist): MNIST dataset in binary format as downloaded from [yann.lecun.com](http://yann.lecun.com/exdb/mnist/);
 * [createdataset.lua](02-Preparing-a-dataset/createdataset.lua): code for serializing the MNIST dataset into `.t7` files and generating samples (section 2.3);
 * [dataloader.lua](02-Preparing-a-dataset/dataloader.lua): code for listing 2.1, 2.2, 2.3 and 2.5. Defines the `DataLoader` and `TensorLoader` classes);
 * [iteratedataset.lua](02-Preparing-a-dataset/iteratedataset.lua): code for listing 2.5. This script tests the `dataloader.lua` file by iterating through it. Only works if `createdataset.lua` was executed before hand;
 * [getmnistsample.lua](02-Preparing-a-dataset/getmnistsample.lua): script for generating MNIST samples consolidated as a single image (used to generate figure 2.1);

[Chapter 3](03-Training-simple-neural-networks): Training simple neural networks
 * [trainlogreg.lua](03-Training-simple-neural-networks/trainlogreg.lua): Training script for applying binary logistic regression on OR dataset. The model is trained using stochastic gradient descent (listing 3.1);
 * [logreg.log](03-Training-simple-neural-networks/logreg.log): log file created by running `th trainlogreg.lua > logreg.log`;
 * [trainlogreg-mnist.lua](03-Training-simple-neural-networks/trainlogreg-mnist.lua): Script for training a multinomial logistic regression model (saved as `logreg-mnist.t7`) using SGD on the MNIST dataset. Training stops after 200 epochs where each epoch consists of 10000 samples divided into mini-batches of 32 random samples, or reaching an estimated empirical risk lower than 0.007, whichever comes first. The resulting model is evaluated on the entire training set of 50000 samples and saved to disk (listing 3.2);
 * [logreg-mnist.log](03-Training-simple-neural-networks/logreg-mnist.log): log file created by running `th trainlogreg-mnist.lua > logreg-mnist.log`. The data can be used to generate a learning curve. Open the file from your favorite spreadsheet application (Microsoft Excel, LibreOffice Calc, etc.) and specify that values are separated by semicolons;
 * [backward.lua](03-Training-simple-neural-networks/backward.lua): demonstrates gradient descent through a criterion. Using the input as a parameter, the loss is minized by tacking a step in opposite direction of gradient (section 8.1.3);
