 Fashion-MNIST Classification using PyTorch

 Project Overview

This project implements a custom neural network using PyTorch for Fashion-MNIST image classification.

The architecture consists of:

* Input Layer
* Flatten Layer
* Hidden Layer
* Branch A:

  * Hidden Layer
  * Hidden Layer
  * Skip Connection
* Branch B:

  * Hidden Layer
  * Hidden Layer
* Concatenation Layer
* Output Layer

 Dataset

Fashion-MNIST Dataset

* 60,000 training images
* 10,000 test images
* 10 clothing categories
* Image size: 28 × 28

 Libraries Used

* PyTorch
* NumPy
* Pandas
* Matplotlib
* Pickle

 Training Results

* Training Accuracy: ~93%
* Validation Accuracy: ~88%

 Project Outputs

* model.pkl
* submission.csv
* loss_accuracy_plot.png

 How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Generate predictions:

python predict.py

 Repository Structure

models/
saved_models/
outputs/
train.py
test.py
predict.py
utils.py
README.md
