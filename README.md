<img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/MaskPatrol.png" width="200"/>

# Mask Patrol

#### A Face Mask Detection System with Deep Learning and Computer Vision

Deep Learning Project

## Goal

The goal of this project is to build a face mask detection system using deep learning algorithsm and computer vision. The project uses the face mask detection images dataset from Kaggle that contains 11,800 images of face with mask and without mask, evenly distributed. The binary classification model was built with Keras/Tensorflow using CNN algorithm and had an accuracy score of 0.99. Using Haar Cascade Classifier as face detector, the face mask detection applies the binary CNN model as mask detector to the area of face and shows the prediction result. The application is built with Streamlit and deployed on Heroku

To learn more, see my [blog post](https://crystalhuang-ds.medium.com/face-mask-detection-with-deep-learning-and-computer-vision-94a965806ab3) and [presentation slides](https://github.com/crystal-ctrl/deep_learning_project/blob/main/presentation.pdf).

## Data

- Face Mask Detection Images Dataset from [Kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) that contains 11,800 images of face with mask and without mask

**Added more datasets for 3-class model**

- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) dataset that contains images of faces with a correctly or incorrectly worn mask (used 8,990 images of incorrectly worn mask and 3,900 images of correctly worn mask)
- CelebFaces Attributes Dataset from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset) that contains images of celebrity faces (used 3,900 images for without mask)

## Workflow

- Code (in [Workflow Folder](https://github.com/crystal-ctrl/deep_learning_project/tree/main/workflow))
  - Streamlit app on Heroku
    - [main python file](https://github.com/crystal-ctrl/deep_learning_project/blob/main/app.py)
    - [Procfile](https://github.com/crystal-ctrl/deep_learning_project/blob/main/Procfile), [setup doc](https://github.com/crystal-ctrl/deep_learning_project/blob/main/setup.sh), [required libray](https://github.com/crystal-ctrl/deep_learning_project/blob/main/requirements.txt) for Heroku
    - [Haar Cascade file](https://github.com/crystal-ctrl/deep_learning_project/blob/main/haarcascade_frontalface_default.xml) for face detection
    - [CNN model](https://github.com/crystal-ctrl/deep_learning_project/blob/main/binary_model.h5) for mask detection

## Results

**Face Mask Detection Result**

![](https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/detected.png)

**Application on Streamlit**

*Face mask detection on Images*

<img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/app%20demo.png" width="900"/>

*Face mask detection on Webcam*

<img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/demo%201.gif" width="900"/>

The app is deployed on Heroku without webcam feature. 

Try out the Heroku app for Mask Patrol [here](https://mask-patrol.herokuapp.com/)!

## Technologies

- Python (pandas, numpy)
- Google colab (Cloud computing)
- Keras/Tensorflow
- OpenCV
- Matplotlib, seaborn
- Streamlit
- Heroku
