from tensorflow.keras.models import load_model
from data_augmentation import valid_generator, test_generator

model = load_model("model_weights/vgg19_final.h5")

val_eval = model.evaluate(valid_generator)
test_eval = model.evaluate(test_generator)

print(f"Validation Loss: {val_eval[0]}, Validation Accuracy: {val_eval[1]}")
print(f"Test Loss: {test_eval[0]}, Test Accuracy: {test_eval[1]}")
