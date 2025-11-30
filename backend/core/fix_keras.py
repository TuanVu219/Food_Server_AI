import tensorflow as tf
import os

# 1️⃣ Định nghĩa lại class
@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_map = self.conv(concat)
        return inputs * spatial_map

# 2️⃣ Chỉ định đường dẫn chính xác
old_path = r"D:\Food_Detection\Food_Recognition\Kanji_Server\backend\core\my_checkpoints\resnet50v2_new.keras"
new_path = r"D:\Food_Detection\Food_Recognition\Kanji_Server\backend\core\my_checkpoints\resnet50v2_new_v2.keras"

# 3️⃣ Kiểm tra file có tồn tại không
if not os.path.exists(old_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file model: {old_path}")

# 4️⃣ Load và re-save
model = tf.keras.models.load_model(
    old_path,
    compile=False,
    custom_objects={"SpatialAttention": SpatialAttention}
)

model.save(new_path)
print("✅ Model đã được re-save thành công với metadata mới!")
print("📁 Saved file:", new_path)
