import pickle
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the encoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to rename multiple files
def main():
    i = 0
    path = r'.\\'
    for filename in os.listdir(path):
        my_dest = "CL" + str(i) + ".wav"
        my_source = path + filename
        my_dest = path + my_dest
        # rename() function will
        # rename all the files
        os.rename(my_source, my_dest)
        i += 1

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
