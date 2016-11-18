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

[Chapter 4](04-Generalizing-deep-neural-networks): Generalizing deep neural networks
 * [tanh.xlsx](04-Generalizing-deep-neural-networks/xor-mlp.xlsx): plot of the hyperbolic tangent activation function (figure 4.2);
 * [trainmlp-xor.lua](04-Generalizing-deep-neural-networks/trainmlp-xor.lua): script for training an MLP with one hidden layer composed of 2 units on the XOR dataset. Used to generate `xor-mlp.xlsx` and figure 4.3;
 * [xor-mlp.xlsx](04-Generalizing-deep-neural-networks/xor-mlp.xlsx): diagram outlining the boundaries of an MLP trained on the XOR dataset (figure 4.3);
 * [overfitting.xlsx](04-Generalizing-deep-neural-networks/overfitting.xlsx): contains learing curve and model overfitting example (figure 4.4 and 4.5);
 * [trainmlp-mnist.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist.lua): upgrades [trainlogreg-mnist.lua](03-Training-simple-neural-networks/trainlogreg-mnist.lua) by moving the definition of hyper-parameters to the cmd-line (listing 4.1 and 4.2).
 * [trainmlp-mnist-crossvalidate.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-crossvalidate.lua): upgrades [trainmlp-mnist.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist.lua) with cross-validation (listing 4.3);
 * [trainmlp-mnist-earlystop.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-earlystop.lua): upgrades [trainmlp-mnist-crossvalidate.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-crossvalidate.lua) with early-stopping (listing 4.4);
 * [trainmlp-mnist-weightdecay.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-weightdecay.lua): upgrades [trainmlp-mnist-earlystop.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-earlystop.lua) with weight decay regularization (listing 4.5);
 * [trainmlp-mnist-hyperopt.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-hyperopt.lua): upgrades [trainmlp-mnist-weightdecay.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-weightdecay.lua) to facilitate hyper-parameter optimization (listing 4.6);
 * [hyperopt-mnist.xlsx](04-Generalizing-deep-neural-networks/hyperopt-mnist.xlsx): spreadsheet used to hyper-optimize the [trainmlp-mnist-hyperopt.lua](04-Generalizing-deep-neural-networks/trainmlp-mnist-hyperopt.lua) script (figure 4.7 and 4.8);
 * [relu.xlsx](04-Generalizing-deep-neural-networks/relu.xlsx): plot of the rectified linear unit (figure 4.9).
