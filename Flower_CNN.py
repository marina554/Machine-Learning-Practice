import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# --- Data folder ---
# Folder structure: FlowerImages/class_name/images
base_dir = "FlowerImages"

# --- Data preprocessing (augmentation + normalization) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # Random rotation
    width_shift_range=0.1,    # Horizontal shift
    height_shift_range=0.1,   # Vertical shift
    shear_range=0.1,          # Shear transformation
    zoom_range=0.1,           # Zoom
    horizontal_flip=True,     # Horizontal flip
    validation_split=0.2      # Use 20% of the data for validation
)

# Validation generator (no augmentation)
valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # Use as training data
)

# Validation data
valid_generator = valid_datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=16,
    class_mode='categorical',
    shuffle=False,
    subset='validation'  # Use as validation data
)

# --- CNN model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# --- Compile ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train ---
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=10
)

print("Training completed!")

# --- Plot training curves ---
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='valid_accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='valid_loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
