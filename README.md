# Binary_Image_Classifier_CNN

## About
In this notebook we are using a subset CIFAR100 dataset, to create a a binary classifier to differentiate the given image has a human or not using Python3 and Google Colaboratory.

## Contents
- Downloading the data
- Exploring and Preprocessing
- Normalizing
- Model
- Better Model (VGG - 16 Architecture)
- Prediction

### Downloading the data



### Exploring and Preprocessing
- Originally, the images have lables such as 'people', 'fish', etc which we can not use.
- Hence, we will convert the lables to 'Human' and 'Not Human'.

### Normalizing
- Convert image features to a 0-1 scale, from a 0-255 scale, by dividing with the maximum (255)
- Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution.     This makes convergence faster while training the network.

### Model
- Convolution Layer
  A matrix of values, called weights, that are trained to detect specific features. Also called a filter, it moves over each part of the   image to check if the feature it is meant to detect is present. To provide a value representing how confident it is that a specific     feature is present, the filter carries out a convolution operation, which is an element-wise product and sum between two matrices.
  
  The output of the convolution operation between the filter and the input image is summed with a bias term and passed through a non-     linear activation function. The purpose of the activation function is to introduce non-linearity into our network.
  
  ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x).

- Max Pooling Layer
  In max pooling, a window passes over an image according to a set stride (how many units to move on each pass). At each step, the         maximum value within the window is pooled into an output matrix, hence the name max pooling.
  
  The result is a downsampled matrix of the image, that still contains the image's characteristics. This helps reduce overfitting and     computational power required. 
  
- Flattening Layer
  Flattening involves transforming the entire pooled feature map matrix into a single long feature vector which is then fed to the         neural network for processing.
  
- Fully connected Layer
  Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks.
  
  Fully connected layer looks at the output of the previous layer (which as we remember should represent the activation maps of high       level features) and determines which features most correlate to a particular class.
  
- I have used batch size of 128 and 10 epochs, which resulted in validation accuracy of 0.8835.  

### Better Model (VGG - 16 Architecture)
- Convolutional neural networks (CNN) have typically had a standard structure - Stacked convolutional layers (optionally followed by       contrast normalization and maxpooling) are followed by one or more fully-connected layers.
- Variants of this basic design are prevalent in the image classification literature and have yielded the best results to-date on MNIST,   CIFAR and most notably on the ImageNet classification challenge.

![alt text](https://github.com/pranavrajwade/Binary_Image_Classifier_CNN/blob/master/VGG%20-%2016%20Architecture.png)


- Dropout: Randomly drop units (along with their connections) from the neural network during training. The reduction in number of         parameters in each step of training has effect of regularization, preventing overfitting.
