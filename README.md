# Face_Toon

## Overview
FaceToon is a machine learning project that leverages the power of Python, Numpy, Tensorflow, and CycleGANs to convert real-life images into a user-customizable cartoon style. This project represents a fusion of artistic creativity and cutting-edge technology, offering users a unique way to transform their photos into cartoon artworks.

## Key Features
Machine Learning Model: Developed a machine learning model using Python, Numpy, and Tensorflow to facilitate the transformation of real-life images into various cartoon styles.
Style Customization: Optimized the neural transfer code to allow users to choose from multiple cartoon styles and merge them with their photos, providing a high level of customization.
CycleGAN Integration: Incorporated CycleGANs into the model, enabling it to generate entirely new cartoon styles that were not present in the original dataset. This addition enhances the variety of artistic possibilities for users.

## Technologies Used
* Python: The primary programming language for building the machine learning model and the application.
* Numpy: Utilized for numerical operations and data manipulation.
* Tensorflow: Employed for developing and training the machine learning model.
* CycleGANs: Integrated CycleGANs to expand the range of available cartoon styles.

## Data Processing
Two datasets were collected for this model. The first was of human faces and the second was of anime faces. The team decided to select anime faces as the cartoon to be applied for this project.
The datasets were cleaned in order to produce the desired outcome from our model. This meant that the images had to have the face centered and clear of any obstructions as well as its size reduced to 64x64 pixels. Since the model used GAN’s, a validation set was not required and thus the data was split with a 80:20 ratio of training and testing.

## Baseline
The baseline model used was called style transfer. We used a pretrained 19-layer VGG (Visual Geometry Group) network available on Pytorch, which simply took two images and combined their styles in the output. However, the problem with this model was that it overlaid the two images onto each other without considering the position of the features; as a result a valid output wouldn't be expected from this model. Nonetheless, we were able to narrow our requirements from a model that converted faces to cartoons to a model that would be able to convert some features of a human face to cartoons.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="Capture (26).png" width="700" alt="Image 1">

## CycleGAN
CycleGAN incorporated two sets of unique generators and discriminators. Generator A  converted anime faces to human faces and was then checked by Discriminator A. Similarly, Generator B and Discriminator B performed the same operation but this time producing anime faces. Finally, these two pairs of generators and discriminators were combined by running a real anime image through Generator A and then through Generator B. After that, MSE (Mean Squared Error) Loss was applied to the real anime image. The idea was that if we could get the same image by converting it twice with our generator, then the style of the image space is learnt and the features of the image are kept  (hair colour, eye shape etc). 

## Training Data Results
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="Capture (27).png" width="700" alt="Image 2">

## Testing Data Results
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="Capture (28).png" width="500" alt="Image 3">
