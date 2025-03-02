import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_augmentation import train_generator, valid_generator

# Load pre-trained VGG19 model
base_model = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(dropout)
output = Dense(2, activation='softmax')(class_2)

model = Model(base_model.inputs, output)

# Compile the model
sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("model_weights/vgg19_model.h5", save_best_only=True, monitor="val_loss")
early_stop = EarlyStopping(monitor="val_loss", patience=4, verbose=1, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=40,
    validation_data=valid_generator,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

if not os.path.isdir('model_weights/'):
    os.mkdir("model_weights/")
model.save("model_weights/vgg19_final.h5", overwrite=True)

print("Model training complete.")
