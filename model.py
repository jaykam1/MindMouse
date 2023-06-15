import pandas as pd
import numpy as np
from PIL import Image
import pyautogui
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate

# Load the DataFrame from the CSV file
df = pd.read_csv('eye_tracking_data.csv')

# Replace paths in df with resized normalised images
left_eye_images, right_eye_images = [], []
for left_eye_image_path, right_eye_image_path in zip(df['left_eye_image_path'], df['right_eye_image_path']):
    
    left_eye_image = Image.open(left_eye_image_path).convert('L').resize((64, 64))
    right_eye_image = Image.open(right_eye_image_path).convert('L').resize((64, 64))
    
    
    left_eye_image_array = np.array(left_eye_image) / 255.0
    right_eye_image_array = np.array(right_eye_image) / 255.0

    left_eye_image_array = np.expand_dims(left_eye_image_array, axis=-1)
    right_eye_image_array = np.expand_dims(right_eye_image_array, axis=-1)
    
    
    left_eye_images.append(left_eye_image_array)
    right_eye_images.append(right_eye_image_array)

# Convert the list of image arrays to a numpy array
left_eye_images = np.array(left_eye_images)
right_eye_images = np.array(right_eye_images)

# Facial landmarks -> np array
landmarks = np.array([np.fromstring(landmarks.strip('[]'), sep=' ') for landmarks in df['landmarks']])

# Normalise mouse coords
screen_width, screen_height = pyautogui.size()
mouse_coords = df[['x', 'y']].values.astype('float64')
mouse_coords[:, 0] /= float(screen_width)
mouse_coords[:, 1] /= float(screen_height)

# Define the eye model
eye_input = Input(shape=(64, 64, 1))
eye_model = Conv2D(32, (3, 3), activation='relu')(eye_input)
eye_model = MaxPooling2D((2, 2))(eye_model)
eye_model = Conv2D(64, (3, 3), activation='relu')(eye_model)
eye_model = MaxPooling2D((2, 2))(eye_model)
eye_model = Flatten()(eye_model)
eye_model = Model(inputs=eye_input, outputs=eye_model)

# Create separate inputs for the left and right eye images
left_eye_input = Input(shape=(64, 64, 1))
right_eye_input = Input(shape=(64, 64, 1))

# Use the eye model for the left and right eye images
left_eye_output = eye_model(left_eye_input)
right_eye_output = eye_model(right_eye_input)

# Define the landmarks model
landmarks_input = Input(shape=(136,))
landmarks_model = Dense(64, activation='relu')(landmarks_input)

# Combine the models
concat = concatenate([left_eye_output, right_eye_output, landmarks_model])

# Add a dense layer and the output layer
output = Dense(64, activation='relu')(concat)
output = Dense(2)(output)

# Define final model
model = Model(inputs=[left_eye_input, right_eye_input, landmarks_input], outputs=output)

# Compile and train model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit([left_eye_images, right_eye_images, landmarks], mouse_coords, epochs=20, validation_split=0.2, shuffle=True)

#Save model
model.save('eye_tracking_modelB4-20.h5')  

