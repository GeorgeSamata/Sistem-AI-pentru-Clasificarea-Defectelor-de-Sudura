import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'optimized_model.keras')

CLASS_NAMES = ['bad_weld', 'crack', 'good_weld', 'porosity', 'spatter']

EXPLICATII = {
    'bad_weld':  'SUDURĂ NECONFORMĂ - Defect de Sudură',
    'crack':     'DEFECT STRUCTURAL - Fisură',
    'good_weld': 'SUDURĂ CONFORMĂ - Sudură de Calitate',
    'porosity':  'DEFECT DE VOLUM - Porozitate',
    'spatter':   'IMPERFECȚIUNI DE SUPRAFAȚĂ - Stropi de Sudură'
}

class WeldingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem AI - Detectie Defecte (Optimizat)")
        self.geometry("800x600")
        ctk.set_appearance_mode("Dark")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.lbl_title = ctk.CTkLabel(self, text="Analiza Automata Suduri - Model Final", font=("Arial", 24, "bold"))
        self.lbl_title.grid(row=0, column=0, pady=20)

        self.lbl_image = ctk.CTkLabel(self, text="Incarca o imagine...", text_color="gray")
        self.lbl_image.grid(row=1, column=0, padx=20, pady=20)
        self.current_image_path = None

        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        self.btn_load = ctk.CTkButton(self.frame_controls, text="📂 Incarca Imagine", command=self.browse_image)
        self.btn_load.pack(side="left", padx=20, pady=20)

        self.btn_predict = ctk.CTkButton(self.frame_controls, text="🔍 Analizeaza", command=self.predict, state="disabled")
        self.btn_predict.pack(side="left", padx=20, pady=20)

        self.lbl_result = ctk.CTkLabel(self.frame_controls, text="Asteptare...", font=("Arial", 18))
        self.lbl_result.pack(side="right", padx=20)

        self.model = None
        self.load_model()

    def load_model(self):
        try:
            print(f"Incarc modelul din: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Fisierul modelului nu exista!")
                
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model incarcat cu succes!")
        except Exception as e:
            self.lbl_title.configure(text=f"Eroare: {str(e)[:50]}...", text_color="red")
            print(f"Eroare Critica: {e}")

    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if filename:
            self.current_image_path = filename
            img = Image.open(filename)
            img = img.resize((400, 300))
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 300))
            self.lbl_image.configure(image=photo, text="")
            self.btn_predict.configure(state="normal")
            self.lbl_result.configure(text="Pregatit", text_color="white")

    def predict(self):
        if not self.model:
            self.lbl_result.configure(text="Eroare: Model lipsa!", text_color="red")
            return
            
        try:
            img = tf.keras.utils.load_img(self.current_image_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_batch)
            score = predictions[0]
            class_index = np.argmax(score)
            
            raw_name = CLASS_NAMES[class_index]
            display_name = EXPLICATII[raw_name] 
            
            confidence = 100 * np.max(score)

            color = "#00FF00" if raw_name == "good_weld" else "#FF5555"
            
            result_text = f"{display_name}\nIncredere: {confidence:.2f}%"
            self.lbl_result.configure(text=result_text, text_color=color)

        except Exception as e:
            self.lbl_result.configure(text="Eroare Analiza", text_color="red")
            print(e)

if __name__ == "__main__":
    app = WeldingApp()
    app.mainloop()