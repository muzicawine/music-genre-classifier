import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Dataset path 
dataset_path = "/kaggle/input/music-genre-classifier/music/audio genre"

# Genre folders 
genres = ['electronic', 'hip hop', 'moombahton', 'pop', 'rock', 'trap']

# Initialize data containers
x = []
y = []
max_len = 130  # Number of MFCC time steps (frames)

# Load and preprocess audio files
for genre in genres:
    folder = os.path.join(dataset_path, genre)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            try:
                # Load 30 seconds of audio
                signal, sr = librosa.load(file_path, duration=30)

                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

                # Normalize MFCCs
                mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

                # Pad or truncate to fixed length
                if mfcc.shape[1] < max_len:
                    pad_width = max_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_len]

                x.append(mfcc)
                y.append(genre)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert lists to NumPy arrays
x = np.array(x)
x = x[..., np.newaxis]  # Add channel dimension
print("Input shape:", x.shape)

# Encode labels and one-hot encode
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)
print("Classes:", label_encoder.classes_)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Define a lightweight CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(genres), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show model structure
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Plot training & validation accuracy and loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
