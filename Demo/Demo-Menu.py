import tkinter as tk
from tkinter import Canvas, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import windnd
import os

model = load_model('C:/Users/makis/Desktop/Visual Plant Disease Detection using Deep Learning, Machine learning and eXplainable AI Techniques/Demo/EfficientNet_model.h5')

def on_drop(files):
    global file_path
    file_path = files[0].decode("utf-8")
    image = Image.open(file_path)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo
    canvas.config(width=photo.width(), height=photo.height())

labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


root = tk.Tk()
canvas = tk.Canvas(root)
canvas.pack()

label = tk.Label(root, text="Drop picture here", width=30, height=5, relief="solid")
label.pack()

windnd.hook_dropfiles(label, func=on_drop)

def on_predict():
    global file_path
    global model
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0])
    prediction = labels[predicted_class]
    label.config(text=f"Prediction: {prediction}")

label = tk.Label(root, text="Prediction: ")
label.pack()

button = tk.Button(root, text="Make Prediction", command=on_predict)
button.pack()

root.mainloop()