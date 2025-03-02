import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load mô hình
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model = Model(base_model.inputs, output)
model.load_weights('model_weights/vgg19_finetuned.h5')

# Danh sách lớp
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 128  # Kích thước ảnh đầu vào

st.title("🩺 Pneumonia Detection App")
st.write("Upload an X-ray image of your lungs and the system will predict whether there are signs of pneumonia.")


# Hàm lấy kết quả

def get_prediction(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0  # Chuẩn hóa pixel
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return CLASS_NAMES[class_idx], confidence


# Upload ảnh
uploaded_file = st.file_uploader("Select lung X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="X-ray photo uploaded", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            label, confidence = get_prediction(img)
            st.success(f"🔹 Result: **{label}**")
            st.write(f"🔢 Reliability: {confidence:.2f}%")
            if label == "PNEUMONIA":
                st.error("⚠️ Warning: Images show signs of pneumonia. Please consult your doctor!!")
            else:
                st.success("✅ No signs of pneumonia were detected..")

st.sidebar.header("⚙️ Guild")
st.sidebar.write("""
1️⃣ Upload lung X-ray images.  
2️⃣ Click the **Predict** button for the system to analyze.  
3️⃣ Prediction results will be displayed immediately.
""")