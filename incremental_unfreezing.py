import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_augmentation import train_generator, valid_generator

# Load lại mô hình đã huấn luyện trước đó
model = load_model("model_weights/vgg19_final.h5")

# Mở khóa một số tầng trong VGG19 để fine-tune
set_trainable = False
for layer in model.layers:
    if layer.name in ['block5_conv3', 'block5_conv4']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Compile lại mô hình sau khi unfreeze
sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("model_weights/vgg19_finetuned.h5", save_best_only=True, monitor="val_loss")
early_stop = EarlyStopping(monitor="val_loss", patience=4, verbose=1, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Fine-tune mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=5,
    validation_data=valid_generator,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Lưu mô hình đã fine-tune
if not os.path.isdir('model_weights/'):
    os.mkdir("model_weights/")
model.save("model_weights/vgg19_finetuned.h5", overwrite=True)

print("Fine-tuning complete.")
