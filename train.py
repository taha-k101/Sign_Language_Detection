from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
import numpy as np
import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
reqd_dir = os.chdir("C:\\Users\\Public\\python\\Projects\\Sign_Language")
print(f"Changed working directory to: {os.getcwd()}")

dir_path = 'C:\\Users\\Public\\python\\Projects\\Sign_Language'
folder_name = 'Sign_Data'
DATA_PATH = os.path.join(dir_path,folder_name) 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

no_sequences = 30
sequence_length = 30

#6. Preprocessing

label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)
# print(np.array(labels).shape)

X = np.array(sequences)
Y = to_categorical(labels).astype(int)
# print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

import numpy as np

# Save training and testing data using NumPy's savez
# dir_path2 = 'C:\\Users\\Public\\python\\Projects\\Sign_Language'
# folder_name2 = 'train_test_data'

# new_dir = os.path.join(dir_path2, folder_name2)
# train_test_loc = os.path.join(new_dir, 'train_test_data.npz')

np.savez('train_test_data.npz', X_train=X_train,X_test=X_test, Y_train=Y_train, Y_test=Y_test)

print("Training and testing data saved successfully!")


#7. Build and Train LSTM Neural Network

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

dir_path1 = 'C:\\Users\\Public\\python\\Projects\\Sign_Language'
folder_name1 = 'Logs'
DATA_PATH1 = os.path.join(dir_path1,folder_name1) 

log_dir = os.path.join(DATA_PATH1)
tb_callback = TensorBoard(log_dir=log_dir)

checkpoint_filepath = os.path.join(DATA_PATH1, 'checkpoints', 'model-{epoch:04d}-{categorical_accuracy:.4f}.keras')
# print()
# print("File path :-")
# print(checkpoint_filepath)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='categorical_accuracy',
    mode='max',
    save_best_only=False)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape = (30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [0.7, 0.2, 0.1]

print(actions[np.argmax(res)])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, epochs = 2000, callbacks = [tb_callback, model_checkpoint_callback])