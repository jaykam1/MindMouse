import cv2
import ctypes
import numpy as np
import time
from PIL import Image
import pyautogui
import dlib
from tensorflow.keras.models import load_model

#Create DS to store moving average of output coordinates for smoother mouse movement
class AverageCoords():
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def add(self, item):
        if len(self.data) > self.window_size:
            self.data.pop(0)
        self.data.append(item)

    def get_average(self):
        return sum(self.data)/len(self.data)
    
# Calculate the Eye Aspect Ratio (EAR)
def get_ear(eye_points):
    
    # Distances between the two sets of vertical eye landmarks 
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])

    # Distance between the horizontal eye landmarks
    c = np.linalg.norm(eye_points[0] - eye_points[3])

    # Calculate the EAR
    ear = (a + b) / (2.0 * c)
    return ear

# Function to check if Caps Lock is on
def is_capslock_on():
    hllDll = ctypes.WinDLL ("User32.dll")
    VK_CAPITAL = 0x14
    return hllDll.GetKeyState(VK_CAPITAL)

def main():
    #Need this to access some buttons e.g close window
    pyautogui.FAILSAFE = False

    # Load the saved model
    model = load_model('eye_tracking_modelB4-20.h5')

    screen_width, screen_height = pyautogui.size()

    #Set up face and eye detectors
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    x_avg = AverageCoords(5)
    y_avg = AverageCoords(5)
    while True:

        ret, frame = cap.read()

        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            if not is_capslock_on():
                for face in faces:
                    # Predict facial landmarks
                    landmarks = predictor(gray, face)

                    # Extract eye images
                    left_eye = gray[landmarks.part(37).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x]
                    right_eye = gray[landmarks.part(44).y:landmarks.part(47).y, landmarks.part(42).x:landmarks.part(45).x]

                    left_landmarks = [landmarks.part(n).x for n in range(36, 42)]
                    right_landmarks = [landmarks.part(n).x for n in range(42, 48)]

                    left_ear = get_ear(left_landmarks)
                    right_ear = get_ear(right_landmarks)

                    ear = (left_ear + right_ear) / 2.0

                    if ear < 0.015:
                        pyautogui.click()

                    left_eye = cv2.resize(left_eye, (64,64))
                    right_eye = cv2.resize(right_eye, (64,64))

                    left_eye = left_eye / 255.0
                    right_eye = right_eye / 255.0

                    left_eye = np.expand_dims(np.expand_dims(left_eye, axis=-1), axis=0)
                    right_eye = np.expand_dims(np.expand_dims(right_eye, axis=-1), axis=0)

                    # Extract facial landmarks
                    landmarks_array = np.array([(point.x, point.y) for point in landmarks.parts()])
                    landmarks_features = np.expand_dims(landmarks_array.flatten(), axis=0)


                    normalised_coords = model.predict([left_eye, right_eye, landmarks_features])
                    coords = normalised_coords * [screen_width, screen_height]
                    x_avg.add(coords[0][0])
                    y_avg.add(coords[0][1])
                    x_coord = x_avg.get_average()
                    y_coord = y_avg.get_average()
                    pyautogui.moveTo(x_coord, y_coord)
    
            # Display the frame
            cv2.imshow('Frame', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
