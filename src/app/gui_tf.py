import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import sys
import os

# Setam calea pentru importuri
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurare UI
st.set_page_config(page_title="SIA Sudura - TensorFlow", layout="wide")
st.title("🏭 SIA - Clasificare Defecte Sudura (TensorFlow)")

# Incarcare Model
MODEL_PATH = 'models/welding_model_v1.keras'
CLASSES = ["Defect (Bad Weld)", "OK (Good Weld)"]

@st.cache_resource
def load_tf_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Modelul nu a fost gasit. Se foloseste un model neantrenat temporar.")
        # Daca nu exista, il cream pe loc (doar structura)
        from src.neural_network.cnn_model import WeldingCNN
        net = WeldingCNN()
        net.save_model(MODEL_PATH)
        return tf.keras.models.load_model(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_tf_model()
    st.sidebar.success("Model TensorFlow incarcat!")
except Exception as e:
    st.sidebar.error(f"Eroare incarcare model: {e}")

# Upload
uploaded_file = st.file_uploader("Incarca Imagine Sudura", type=['jpg', 'png', 'jpeg'])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Procesare imagine
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.image(image_rgb, caption="Imagine Originala", use_container_width=True)
        
    if st.button("Analizeaza"):
        with st.spinner("Procesare TensorFlow..."):
            # Preprocesare specifica TF (Resize 224x224)
            img_resized = cv2.resize(image_rgb, (224, 224))
            img_array = tf.expand_dims(img_resized, 0) # Batch dimension
            
            # Inferenta
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_id = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) # Simplificat pt demo
            
            # Afisare
            label = CLASSES[class_id] if class_id < len(CLASSES) else "Necunoscut"
            
            with col2:
                st.subheader("Rezultat Analiza:")
                if "Defect" in label:
                    st.error(f"REZULTAT: {label}")
                else:
                    st.success(f"REZULTAT: {label}")
                    
                st.progress(float(confidence))
                st.write(f"Incredere Model: {confidence*100:.2f}%")
                st.code(f"Raw Output Tensor: {predictions}")