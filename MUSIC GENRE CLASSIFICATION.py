import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical                                       #type:ignore
from tensorflow.keras.models import Sequential                                         #type:ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout      #type:ignore

# Path to your dataset
dataset_path = r"C:\Users\satya\Desktop\jupyter projects\music"
genres = ['electronic', 'hiphop', 'pop', 'moombahton', 'rock', 'trap']

x = []
y = []
max_len = 130  # Adjust this if needed depending on MFCC time frames

# Load and preprocess the data
for genre in genres:
    folder = os.path.join(dataset_path, genre)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            try:
                signal, sr = librosa.load(file_path, duration=30)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
                if mfcc.shape[1] < max_len:
                    pad_width = max_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_len]

                x.append(mfcc)
                y.append(genre)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to NumPy arrays
x = np.array(x)
x = x[..., np.newaxis]  # Add channel dimension
print("Input shape:", x.shape)

# Encode labels and convert to one-hot
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("Classes:", label_encoder.classes_)
y = to_categorical(y)

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(genres), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary
model.summary()

# Train
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
