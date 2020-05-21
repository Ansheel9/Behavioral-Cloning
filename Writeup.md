---
title: 'Writeup'
disqus: Ansheel Banthia
---

Project 4 - Behavioral Cloning :oncoming_automobile: 
===

Udacity Self-Driving Car Engineer Nanodegree

## Table of Contents

 - [ Summary ](#sum)
 - [ Project Structure ](#des)
 - [ Data collection and balancing ](#coll)
 - [ Data Augmentation and preprocessing ](#aug)
 - [ Model Building ](#mod)
 - [ Result ](#res)
 - [ Improvement to Pipeline ](#fut)

<a name="sum"></a>
## Summary

Track 1 | Track 2
------------|---------------
![training_img](./images/track_one.gif) | ![validation_img](./images/track_two.gif)

In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

The goal of this project was to train a end-to-end deep learning model that would let a car drive itself around the track in a driving simulator.

<a name="des"></a>
## Project structure

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Implements model architecture and runs the training pipeline.                      |
| `model.h5`                   | Model weights.                                                                     |
| `drive.py`                   | Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions based on real-time data simulator app is sending. |
| `video.py`                   | Creates a video based on images found in the `data` directory. |

<a name="coll"></a>
## Data collection and balancing

The provided driving simulator had two different tracks. One of them was used for collecting training data, and the other one — never seen by the model — as a substitute for test set.

The driving simulator would save frames from three front-facing "cameras", recording data from the car's point of view; as well as various driving statistics like throttle, speed and steering angle. We are going to use camera data as model input and expect it to predict the steering angle in the `[-1, 1]` range.

I have collected a dataset containing approximately **1 hour worth of driving data** around track 1. This would contain both driving in _"smooth"_ mode (staying right in the middle of the road for the whole lap), and _"recovery"_ mode (letting the car drive off center and then interfering to steer it back in the middle). 

<a name="aug"></a>
## Data Augmentation and preprocessing

A single frame in the video gave three outputs: images from the camera mounted on the left, center and right side of the vehicle. To augment the center camera image data with left and right camera image data, I used a correction factor for the steering wheel. So, for the same frame, the steering angle for the left, center and right camera image will be:

Steering angle for the left camera image:`(Center_Steering_angle + Correction_factor)`

Steering angle for the center camera image:`(Center_Steering_angle)`

Steering angle for the right camera image:`(Center_Steering_angle - Correction_factor)`

Also, to further enlarge the dataset, the images were flipped in order to add driving behavior for the right turn as the track only had left turns. The steering angle for the flipped image will be:

Steering angle for the normal camera image: (Steering_angle)

Steering angle for the flipped camera image: (-Steering_angle)

After the collection process, I had total of **1 hour worth of images in the dataset** . I then preprocessed this data by normalizing and cropping the images. Then I finally randomly shuffled the data set and put 20% of the recorded data into the validation set.

Left Camera Image | Centre Camera Image | Right Camera Image
------------|--------------- |---------------
![training_img](./images/track_one.gif) | ![validation_img](./images/track_two.gif) | ![validation_img](./images/track_two.gif)

Flipped Image | Cropped
------------|---------------
![training_img](./images/track_one.gif) | ![validation_img](./images/track_two.gif)

<a name="mod"></a>
## Model Building

I started by looking at some existing works on self driving cars, including,
- [Udacity self-driving-car project](https://github.com/udacity/self-driving-car)
- [DeepDrive](http://deepdrive.io/)
- [commma.ai research](https://github.com/commaai/research)
- [Nvidia Paper: End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)

I am interested in trying different pipeline that are choices at multiple steps, e.g., data generation/augmentation, image/output preprocessing, and modelling. So the next step for me was to experiment with different structure from these works. I thought Xception model might be appropriate because it uses an end-to-end approach which means it uses minimum training data to learn to steer around the road.

![training_img](./images/track_one.gif)

Xception pretrained model (on ImageNet) and 1 global pooling layer and 3 dense layers as new head.
<a name="short"></a>

<a name="res"></a>
## Result
---
I used combined dataset generated by me as well as published by the Udacity. The model developed using that dataset works well on both tracks as shown.

Track 1 | Track 2
------------|---------------
![training_img](./images/track_one.gif) | ![validation_img](./images/track_two.gif)

<a name="fut"></a>
## Improvement to Pipeline
---
When it comes to extensions and future directions, I would like to highlight followings.
* Train a model in real road conditions. For this, we might need to find a new simulator.
* Experiment with other possible data augmentation techniques.
* When we are driving a car, our actions such as changing steering angles and applying brakes are not just based on instantaneous driving decisions. In fact, curent driving decision is based on what was traffic/road condition in fast few seconds. Hence, it would be really interesting to seee how Recurrent Neural Network (RNN) model such as LSTM perform this problem.
* Finally, training a (deep) reinforcement agent would also be an interesting additional project.


