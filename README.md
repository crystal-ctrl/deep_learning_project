# <img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/MaskPatrol.png" width="200"/>Mask Patrol

#### A Face Mask Detection System with Deep Learning and Computer Vision

Deep Learning Project

## Abstract

The goal of this project is to build a face mask detection system using deep learning algorithsm and computer vision. The project uses the face mask detection images dataset from Kaggle that contains 11,800 images of face with mask and without mask, evenly distributed. The binary classification model was built with Keras/Tensorflow using CNN algorithm and had an accuracy score of 0.99. Using Haar Cascade Classifier as face detector, the face mask detection applies the binary CNN model as mask detector to the area of face and shows the prediction result. The application is built with Streamlit and deployed on Heroku



## Backstory

It's the year of 2021, we all learned the importance of mask in our lives. As life slowly returns to normal, there are increased crowd volume in the public space. Although some states have lifted mask requirements, it is still mandated at indoor public places like airports and hospitals. But it's difficult and inefficient to inspect the large crowd with labor screening. So, the goal of this project is to build a face mask detection system using deep learning algorithms and computer vision to let machine help with inspection. 

## Data

- Face Mask Detection Images Dataset from [Kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) that contains 11,800 images of face with mask and without mask

**Added more datasets for 3-class model**

- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) dataset that contains images of faces with a correctly or incorrectly worn mask (used 8,990 images of incorrectly worn mask and 3,900 images of correctly worn mask)
- CelebFaces Attributes Dataset from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset) that contains images of celebrity faces (used 3,900 images for without mask)

## Methodology

**EDA**

The data is pretty balanced between the two classes (about 50/50)

#### Train Mask Detector

***Binary Classification Model***

1. Extracted small subset of data from the full dataset
2. Started with small subset of data to build work pipeline using CNN
3. Used the full dataset on the CNN model pipeline
4. Evaluated CNN model
5. Made prediction on dataset and new images

***3-Class Model***

1. Add more datasets
2. Extracted small subset of data from the full dataset
3. Started with small subset of data to build work pipeline using CNN
4. Used the full dataset on the CNN model pipeline
5. Evaluated CNN model
6. Made prediction on dataset and new images

#### Detect Faces (with Haar Cascade Classifier) and Apply Mask Detector

1. Load mask detector (Binary classification model)
2. Load images / start webcam video stream
3. Detect faces in image / video stream with Haar Cascade Classifier
4. Extract each face area
5. Apply mask detector to each face area
6. Show prediction results

## Findings

**Binary Classification Model**

- Accuracy: 0.99
- Precision: 0.99
- Recall: 0.99
- F1 score: 0.99

**3-Class Model**

- Accuracy: 0.99
- Precision: 0.99
- Recall: 0.99
- F1 score: 0.99

The binary model had great performance both within and outside the dataset while 3-class model did good within the dataset but not so good labeling "incorrect mask" in new images. So the binary model is more suitable for real-life images and is used as the mask detector in the application.

**Face Mask Detection Result**

![](https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/detected.png)

**Application on Streamlit**

*Face mask detection on Images*

<img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/app%20demo.png" width="900"/>

*Face mask detection on Webcam*

<img src="https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/demo%201.gif" width="900"/>

The [app](https://mask-patrol.herokuapp.com/) is deployed on Heroku without webcam feature. 

## Workflow

- Code (in [Workflow Folder](https://github.com/crystal-ctrl/deep_learning_project/tree/main/workflow))
  - Streamlit app on Heroku
    - [main python file](https://github.com/crystal-ctrl/deep_learning_project/blob/main/app.py)
    - [Procfile](https://github.com/crystal-ctrl/deep_learning_project/blob/main/Procfile), [setup doc](https://github.com/crystal-ctrl/deep_learning_project/blob/main/setup.sh), [required libray](https://github.com/crystal-ctrl/deep_learning_project/blob/main/requirements.txt) for Heroku
    - [Haar Cascade file](https://github.com/crystal-ctrl/deep_learning_project/blob/main/haarcascade_frontalface_default.xml) for face detection
    - [CNN model](https://github.com/crystal-ctrl/deep_learning_project/blob/main/binary_model.h5) for mask detection

## Technologies

- Python (pandas, numpy)
- Google colab (Cloud computing)
- Keras/Tensorflow
- OpenCV
- Matplotlib, seaborn
- Streamlit
- Heroku

## Communication

Try out the Heroku app for Mask Patrol [here](https://mask-patrol.herokuapp.com/)!

To learn more, see my blog post and [presentation slides](https://github.com/crystal-ctrl/deep_learning_project/blob/main/presentation.pdf).