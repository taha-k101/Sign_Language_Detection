import tensorflow
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow import keras
from keras import models
import os

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable 
    results = model.process(image)                 # Make prediction 
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
                             
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

dir_path = r'C:\Users\Home\OneDrive\Desktop\PROJECTS\ppMP\GitHub\Sign-Language_Detection\Logs\checkpoints'
file_name = 'model-0420-0.9176.keras'
model_path = os.path.join(dir_path, file_name)

model = models.load_model(model_path)
actions = ['hello', 'thanks', 'iloveyou']

sequence = []
sentence = []
threshold = 0.4


import numpy as np

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
reqd_dir = os.chdir(r"C:\Users\Home\OneDrive\Desktop\PROJECTS\ppMP\GitHub\Sign-Language_Detection")
print(f"Changed working directory to: {os.getcwd()}")

# Load the data from the NumPy archive
data = np.load('train_test_data.npz')

# Access the split data using their names in the archive
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

print("Training and testing data loaded successfully!")

res = model.predict(X_test)
res = [1]

import time

def visualize():
    if res[np.argmax(res)] > threshold:
        if len(sentence) > 0:
            if actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
        
        else:
           sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5:
       sentence = sentence[-5:]

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(0) #grabbing webcam
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

  while cap.isOpened():     #looping thru frames to make it a video

    #Read feed
    ret, frame = cap.read()

    #make detections
    image,  results = mediapipe_detection(frame, holistic)
    print(results)

    #draw landmarks
    draw_styled_landmarks(image, results)

    # 2. Prediction Logic

    keypoints = extract_keypoints(results)
    # sequence.append(keypoints)
    # sequence = sequence[-30:]
    sequence.insert(0, keypoints)
    sequence = sequence[:30]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)]) 


    # 3. Visualization Logic

    if res[np.argmax(res)] > threshold:
        if len(sentence) > 0:
            if actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
        
        else:
           sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5:
       sentence = sentence[-5:]

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #show to screen
    cv2.imshow('OpenCV Feed', image)

    #breaking gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
print(res)
# print(res[np.argmax(res)])