import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import json

MODEL_SAVE_PATH = 'model_car_brand.h5'
CLASS_INDICES_PATH = 'class_indices.json'
IMAGE_SIZE = (224, 224)


def predict_brand_gui(image_path):
    try:
        model = load_model(MODEL_SAVE_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = img_array[np.newaxis, ...]
        predictions = model.predict(img_array)
        if len(predictions[0]) == 1:
            predicted_class = int(predictions[0][0] > 0.5)
        else:
            predicted_class = predictions[0].argmax()
        return idx_to_class[predicted_class]
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się przewidzieć marki: {e}")
        return None


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            img = Image.open(file_path)
            frame_right.update_idletasks()
            frame_width = frame_right.winfo_width()
            frame_height = frame_right.winfo_height()
            img = img.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            root.image_path = file_path
            result_label.config(text="Obraz wczytany. Kliknij 'Rozpoznaj markę'.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się otworzyć obrazu: {e}")


def recognize_brand():
    if hasattr(root, 'image_path') and root.image_path:
        predicted_brand = predict_brand_gui(root.image_path)
        if predicted_brand:
            result_label.config(text=f"Przewidywana marka: {predicted_brand}")
    else:
        messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz.")


root = tk.Tk()
root.title("System Rozpoznawania Marek Pojazdów")

frame_left = tk.Frame(root, width=300, bg="white")
frame_left.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

frame_right = tk.Frame(root, bg="lightgray")
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

title_label = tk.Label(frame_left, text="System Rozpoznawania Marek Pojazdów", font=("Arial", 16), bg="white")
title_label.pack(pady=20)

button_open = tk.Button(frame_left, text="Wczytaj obraz", command=open_image, width=20, bg="blue", fg="white")
button_open.pack(pady=10)

button_recognize = tk.Button(frame_left, text="Rozpoznaj markę", command=recognize_brand, width=20, bg="green",
                             fg="white")
button_recognize.pack(pady=10)

result_label = tk.Label(frame_left, text="Wynik pojawi się tutaj", font=("Arial", 12), bg="white")
result_label.pack(pady=20)

image_label = tk.Label(frame_right, bg="lightgray")
image_label.pack(fill=tk.BOTH, expand=True)

root.mainloop()
