import tensorflow as tf
from tensorflow import keras
from keras import models
import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
reqd_dir = os.chdir("C:\\Users\\Public\\python\\Projects\\Sign_Language")
print(f"Changed working directory to: {os.getcwd()}")

dir_path = 'C:\\Users\\Public\\python\\Projects\\Sign_Language\\Logs\\checkpoints'
file_name = 'model-0420-0.9176.keras' #from prev logs, 0420, 0.9176
model_path = os.path.join(dir_path, file_name)

model = models.load_model(model_path)

import numpy as np

# Load the data from the NumPy archive
data = np.load('train_test_datanpz')

# Access the split data using their names in the archive
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

print("Training and testing data loaded successfully!")

print(model.summary())

res = model.predict(X_test)
actions = ['hello', 'thanks', 'iloveyou']
print(actions[np.argmax(res[1])])
print(actions[np.argmax(Y_test[1])])

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(Y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))