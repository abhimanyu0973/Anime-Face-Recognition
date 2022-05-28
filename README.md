# Anime-Face-Recognition'

## Introduction

Hello there !

This particular project (Anime-Face-Recognition) was built for the Microsoft engage program (https://acehacker.com/microsoft/engage2022/) 
under the theme of face recognition.

The reason why I chose this is because there are already many apps for human face recognition and its fairly common. 
However there are not as many apps for anime character recognition.

My idea was to recognise all the characters in a picture and give details about each character and the anime which they are from.

This can help anime watchers get information about the characters which they come across in posts and videos. 
It can help the people who have already watched the anime but have forgotten about the character or they simply wish to revisit the anime.
It can also help you in an anime quiz. 
This can be even more useful for people who read manga.(in simple terms manga is a Japanese comic book or graphic novel).

The task was to implement this in the form of an application and I decided to go with a web application.

## How was this project built ?

You would think that the same face-recognition api for human face detection could also work for anime face, but it does not work because the facial features of 2D
anime charcaters are quiet different compared to humans.

## Face detection
First task was to seperate all the characters in an a particular image.

This was done using OpenCV cascade classifier lbpcascade_animeface.xml.(made by nagadomi)
It is a face detector for anime characters using OpenCV which is based on LBP cascade.

Now that the faces are sperated we have to get their identity.

## Anime Character recognition
For this I decided to train an Image Classification model Trained on the Resnet9 model and then transfer learning using Resnet34 model(pretrained on the Imagenet dataset).
I decided to use this as I had previously used it before and the results are quiet satisfying.(Was able to get about 90% accuracy on 8 characters)(on validation dataset).

I trained the model on Google Collab . 
Here is the link for the training process (https://colab.research.google.com/drive/1HhEJiovuMoocqCDrelsVbKCCyVIvkJNR?usp=sharing).
The python notebook for the training process (downloaded from the link above) is also present in this repo named : "Model_Training.ipynb"
It has been documented well in case you want to go through the training process in detail.

My aim was to train the model for a higher accuracy and try other models for training as well, but the time was limited as we had a deadline to submit the project.

The Dataset which I used for training my model consists of only 8 characters.(about 100 images for each character)
Link to the dataset (https://www.kaggle.com/datasets/abhimanyugautam/finalmaybe)
This is because it was time taking to gather a large dataset and also the training time would be longer.
Also, once the base project has been made it can be easily expanded with time. So my aim was to get things working first.

The final trained model is present in the repository named "final_maybe.pth"

All set !! Only thing left was to bring everything together in a web application.

## Making a web application

Being completely new to web development this was pretty challenging.
I decided to use the flask framework as I heard it is simple to use. So I could learn it quickly.

I have commented the code for the flask web application. Hopefully it is understandable.

Also I have launched the web app using heroku for free.
Link to the web app : https://animenhk.herokuapp.com/

## Steps to run the application on your device

1. Firstly clone the repository
2. Make a virtual environment in the directory containing all the cloned files.
3. Activate the virtual environment.
4. Install all the dependencies using pip install -r requirements.txt in the command line.
5. type "flask run" in the command line.

### Note
Currently the model runs on cpu, if you want to use the gpu/cuda you will have to make some changes in app.py (read the comments in app.py for the changes)
Also you should have torch with cuda installed(your gpu must support torch with cuda). The current requirements.txt file only has torch with cpu.

I hosted my web application with cpu because torch with cuda is large in size and free membership of heroku does not support that size.



