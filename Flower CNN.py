import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# --- Data folder ---
# Use a relative path so personal information is not exposed
# Make sure to create the folder with the same structure before running
base_dir = "FlowerImages"  # ← Do not include personal info

# Data generator (training only)
train_datagen = ImageDataGenerator(rescale=1./255)

# Training data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=1,           # Small batch size for limited data
    class_mode='categorical',
    shuffle=True            # Shuffle the data during training
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Match the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (no validation)
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Use all images in one epoch
    epochs=5
)

print("Training completed!")

# --- Plot training curves ---
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], marker='o', label='train_accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], marker='o', label='train_loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
