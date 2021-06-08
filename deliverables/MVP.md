# MVP
The goal of this project is to build a face mask detector using deep learning algorithm.

My first model was binary model using CNN and had accuracy score of 0.99. Since the model performed so well already, I decided to change this into a multiclass model by adding another class "incorrect mask" with an additional 5,900 images from [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net). 

After setting up the workflow with a smaller dataset, I built the multi class model with CNN, which has accuracy score of 0.99 for both train and val (as shown in the acc/loss graph below).

![](https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/trainval_multiclass2.png)

The confusion matrix also shows that the model only missed 8 predictions in the test set. 

![](https://github.com/crystal-ctrl/deep_learning_project/blob/main/images/multiclass_cm2.png)

The model performance on unseen test data is:

- Accuracy: 0.99
- Precision: 0.99
- Recall: 0.99
- F1 score: 0.99

For the next step, I will try face detecion using Haar cascades in OpenCV to extract faces from large pictures so the model can run prediction on new images.