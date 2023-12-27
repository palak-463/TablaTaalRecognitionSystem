# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import models, layers
import pickle
import librosa

# Function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=60)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        features.append(np.mean(e))
    
    return features

# Function to create the dataset CSV file
def create_dataset():
    data = []
    genres = ['addhatrital', 'trital', 'ektal','rupak','jhaptal','bhajani','dadra','deepchandi'] 

    for genre in genres:
        folder_path = os.path.join(os.getcwd(), genre)
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                features = extract_features(file_path)
                data.append(features + [genre])

    # Create a DataFrame and save it to CSV
    header = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'] + [f'mfcc{i}' for i in range(1, 21)] + ['label']
    df = pd.DataFrame(data, columns=header)
    df.to_csv('data.csv', index=False)

# Main part of the script
if __name__ == '__main__':
    # Create the dataset CSV file
    create_dataset()

    # Load the dataset
    data = pd.read_csv('data.csv')

    # Check if 'filename' column exists before dropping
    if 'filename' in data.columns:
        data = data.drop(['filename'], axis=1)

    # Encode labels and scale features
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['label'])

    # Check the number of classes (replace 3 with the actual number of tabla tals)
    num_classes = 8

    # Check if labels are within the expected range
    if any(label >= num_classes for label in y):
        raise ValueError("Label values outside the valid range of tabla tals.")

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=512, validation_data=(X_test, y_test))

    # Save the model, encoder, and scaler to files
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Model training and saving completed.")
