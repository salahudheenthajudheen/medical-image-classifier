import json
from pathlib import Path

import numpy as np
import streamlit as st
import keras
from PIL import Image

IMG_SIZE = (224, 224)

st.set_page_config(page_title="Medical Image Classifier", page_icon="🩺")
st.title("🩺 Multi-Disease Medical Image Classifier")
st.warning("Educational prototype only. This result is not a medical diagnosis.")


@st.cache_resource
def load_model(module):
    labels = json.loads(Path("models").joinpath(f"{module}.json").read_text())
    weights_path = Path("models") / f"{module}.weights.h5"
    if weights_path.exists():
        base = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(len(labels), activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.load_weights(weights_path)
    else:
        model = keras.models.load_model(Path("models") / f"{module}.keras", compile=False)
    return model, labels


available = sorted(p.stem for p in Path("models").glob("*.keras"))
if not available:
    st.info("No trained models found. Add a dataset and run train.py first.")
    st.stop()

module = st.selectbox("Choose an image module", available)
uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Analyze image", type="primary"):
        model, labels = load_model(module)
        resized = image.resize(IMG_SIZE)
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
        probabilities = model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        st.subheader(f"Prediction: {labels[index]}")
        st.metric("Confidence", f"{probabilities[index] * 100:.2f}%")
        st.bar_chart({label: float(score) for label, score in zip(labels, probabilities)})
