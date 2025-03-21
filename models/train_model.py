import os
import re
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to the dataset with subfolders "real" and "fake"
data_dir = r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\extracted_frames"

# Define paths for saving model and checkpoints
checkpoint_dir = r"C:\Users\gau68\DeepfakeDetector\models\checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
# Checkpoint path pattern includes epoch number, e.g., cp_01.weights.h5
checkpoint_path_pattern = os.path.join(checkpoint_dir, "cp_{epoch:02d}.weights.h5")

# Data augmentation and generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Setup checkpointing callback to save a checkpoint after each epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_pattern,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

# Determine the initial epoch if any checkpoint exists
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "cp_*.weights.h5"))
initial_epoch = 0
if checkpoint_files:
    # Find the most recent checkpoint file
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print("Resuming training from checkpoint:", latest_checkpoint)
    
    # Extract the epoch number from the filename using a regular expression
    match = re.search(r"cp_(\d+)\.weights\.h5", latest_checkpoint)
    if match:
        initial_epoch = int(match.group(1))
    model.load_weights(latest_checkpoint)

# Set total epochs (for example, 10 epochs total)
total_epochs = 10

# Train the model, resuming from the correct epoch count
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=total_epochs,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback]
)

# Save the final trained model
model.save(r"C:\Users\gau68\DeepfakeDetector\models\deepfake_detector.h5")
