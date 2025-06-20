import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Path to the dataset
dataset_path = "./Quality Dataset/train"

# 2. Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# 3. Better Data Preparation (with image variations)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Images will be rotated a little
    width_shift_range=0.2,  # Images may move left/right
    height_shift_range=0.2, # Images may move up/down
    zoom_range=0.2,         # Images may zoom in/out
    horizontal_flip=True,   # Images may flip sideways
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 4. Better Model Building (like building with more LEGO blocks)
model = Sequential([
    # First set of blocks
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    
    # Second set of blocks
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third set of blocks (new!)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Putting it all together
    Flatten(),
    Dense(256, activation='relu'),  # Bigger thinking layer
    Dropout(0.5),                  # Helps prevent memorizing
    Dense(1, activation='sigmoid') # Final decision
])

# 5. Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 6. Train the model (with patience)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,  # We'll try 30 times to learn
    verbose=1   # Shows progress bars
)

# 7. Save the model
model.save("Fruit_Quality_Classifier.h5")
print("✅ Model saved as Fruit_Quality_Classifier.h5")

# 8. Let's see how well it learned
print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])