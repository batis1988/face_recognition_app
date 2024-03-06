# Face Recognition App

This is how we can build end-to-end face recognition app using open source tools.

Current project consist of 3 parts:

[x] 1. Build / Train ML model

[x] 2. Creating web app

[x] 3. Deploy app

## Developing Face Recognition Model

First of all it’s necessary to report about data quality with EDA. You can reach the results in ___'s notebook. After this, it will be advisable to set up a route for data preprocessing and converting images into features. This made it possible to provide an optimal model architecture for the task of face recognition. Next, the model was built into the pipeline to capture the stages and prepare for creating the application. And finally, Flask API based app was built and deployed.

So, to bring you explicitness of the contest, the main purpose of the project is to detect a person, identify gender and age, emotions (optional)

## Data source

The source of images is a well known `IMDB-WIKI**–500k+ face images with age and gender labels**`, published in 2015. And, to be more precise is the "[Faces only (1 GB)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)" version. 

![Untitled](Face%20Recognition%20App%20144200d230fc432f9ae300f9d41214d1/Untitled.png)

## Preprocessing stage reveal

### Detecting Face

1. Detecting with Haar Cascade CLF (the `*.xlm`  pattern brought by OpenCV’s [FrontalFace_Deafault](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)) 
2. Cropping and alignment
3. Labeling

### Structuring Images

1. Gray scaling
2. Cut off less than 60 dim’s images (according to EDA, provided by notebook’s)
3. Resizing (100x100)

### Extract Features

Please, follow the [Stanford’s Eigenimages course material](https://web.stanford.edu/class/ee368/Handouts/Lectures/2019_Winter/10-EigenImages.pdf) to get known the idea.

1. Extract eigen features
2. Dimensional reduction

## Flask App

We’ll go the convenient way by creating a flask application, that appears in such interface:

1. Home page
2. Application page
3. Gender prediction page
