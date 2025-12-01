import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# -----------------------------
# 1️⃣ Custom layer (bắt buộc phải có trước load model)
# -----------------------------
@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters=1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_map = self.conv(concat)
        return inputs * spatial_map

# -----------------------------
# 2️⃣ Cấu hình
# -----------------------------
IMG_SIZE = 224
CLASS_NAMES = [
    "Banh mi", "Banh xeo", "Bun bo Hue", "Bun dau mam tom",
    "Bun thit nuong", "Cao lau", "Com tam", "Goi cuon",
    "Hu tieu", "Mi quang","Pho"
]

HF_REPO = "TuanVu219/Vit_Checkpoint_New"
HF_CKPT_FILENAME = "resnet50v2_121.h5"

# -----------------------------
# 3️⃣ Download & load model trực tiếp từ HF
# -----------------------------
MODEL_PATH = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT_FILENAME)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully from HF:", MODEL_PATH)

# -----------------------------
# 4️⃣ Hàm tiền xử lý
# -----------------------------
def preprocess_single_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_resized

# -----------------------------
# 5️⃣ Hàm dự đoán
# -----------------------------
def predict_external_image(img_path):
    img_array, img_orig = preprocess_single_image(img_path)
    preds = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    class_name = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else "Unknown"
    return class_name, confidence
