import cv2
import pandas as pd
import pyautogui
import os
import dlib
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a DataFrame to store the data
df = pd.DataFrame(columns=['left_eye_image_path', 'right_eye_image_path', 'landmarks', 'x', 'y'])

# Set up face and eye detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Create a directory to store the eye images
os.makedirs('left_eye_images', exist_ok=True)
os.makedirs('right_eye_images', exist_ok=True)

# Initialize an ID for each image
image_id = 0

# Collect data
while True:
    ret, frame = cap.read()
    
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract eye images
            left_eye = gray[landmarks.part(37).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x]
            right_eye = gray[landmarks.part(44).y:landmarks.part(47).y, landmarks.part(42).x:landmarks.part(45).x]

            # Save the eye images
            left_eye_image_path = f'left_eye_images/{image_id}.png'
            cv2.imwrite(left_eye_image_path, left_eye)
            
            right_eye_image_path = f'right_eye_images/{image_id}.png'
            cv2.imwrite(right_eye_image_path, right_eye)

            # Extract facial landmarks
            landmarks_array = np.array([(point.x, point.y) for point in landmarks.parts()])
            landmarks_features = landmarks_array.flatten()

            # Get the current mouse coordinates
            x, y = pyautogui.position()

            # Append the data to the DataFrame
            df = df.append({
                'left_eye_image_path': left_eye_image_path,
                'right_eye_image_path': right_eye_image_path,
                'landmarks': landmarks_features,
                'x': x, 'y': y
            }, ignore_index=True)
            
            image_id += 1
            
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Break the loop on 'q' qkey press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Save the DataFrame to a CSV file
df.to_csv('eye_tracking_data.csv', index=False)

