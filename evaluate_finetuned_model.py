import tensorflow as tf
from tensorflow.keras.models import load_model
from data_augmentation import valid_generator, test_generator

# Load mô hình đã fine-tune
model = load_model("model_weights/vgg19_finetuned.h5")

# Đánh giá mô hình
val_eval = model.evaluate(valid_generator)
test_eval = model.evaluate(test_generator)

print(f"Validation Loss: {val_eval[0]}, Validation Accuracy: {val_eval[1]}")
print(f"Test Loss: {test_eval[0]}, Test Accuracy: {test_eval[1]}")
