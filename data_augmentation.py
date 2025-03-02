from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,
    shear_range=0.2,
    width_shift_range=0.4,
    height_shift_range=0.4,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    "chest_xray/train",
    batch_size=32,
    target_size=(128, 128),
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

valid_generator = valid_datagen.flow_from_directory(
    "chest_xray/val",
    batch_size=32,
    target_size=(128, 128),
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    "chest_xray/test",
    batch_size=32,
    target_size=(128, 128),
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

if __name__ == "__main__":
    print("Data augmentation setup complete.")
