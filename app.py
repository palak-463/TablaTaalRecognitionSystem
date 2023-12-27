from flask import Flask, request, render_template
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import librosa

app = Flask(__name__)

# Load the saved model, encoder, and scaler using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def extract_features(file_path):
    #here it loads a file and then return the audio time series=y and sampling rate=sr
    y, sr = librosa.load(file_path, mono=True, duration=60) 
    #chroma short time fourier transform for knwoing the pitch classes
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #root mean square it indicates the energy/amplitude of audio signal
    rms = librosa.feature.rms(y=y)
    #represents central of mass and brightness of sound
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    #width of spectral band
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    #represents frequency below which specified percentage of total spectral energy lies
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    #Zero Crossing rate represents the rate at which the audio signal chnages its sign(measures how noisy data is)
    zcr = librosa.feature.zero_crossing_rate(y)
    #mel-frequency ceptral coefficients represents short term power spectrum of sound
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  
    #information about the spectral shape of the audio signal.
    features = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        features.append(np.mean(e))
    
    return features

def predict_tal(file_path):
    features = extract_features(file_path)
    # predict the tabla tal
    prediction = model.predict(scaler.transform([features]))  # Standardize the feature array before prediction
    # Assuming your model outputs probabilities for different tabla tals
    predicted_tal_index = np.argmax(prediction)
    tabla_tals = ['addhatrital', 'trital', 'ektal','rupak','jhaptal','bhajani','dadra','deepchandi'] 
    predicted_tal = tabla_tals[predicted_tal_index]
    return predicted_tal

# Define the endpoint for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the endpoint for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    filename = file.filename
    file_path = os.path.join('static', 'uploads', filename)
    file.save(file_path)
    
    predicted_tal = predict_tal(file_path)
    return render_template('result.html', tal=predicted_tal)

if __name__ == '__main__':
    app.run(debug=True)
