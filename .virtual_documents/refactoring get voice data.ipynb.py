# refactoring get voice data


# this is the playground for fixing mfcc addition

# initialize
from tqdm import tqdm
from time import sleep

import glob
import parselmouth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# needed for mfcc calculation
import statistics
import speechpy
from scipy.io import wavfile











_path = test_path = "/Users/leochoo/dev/VoiceDisorderSVM/data/SVD/test_audio/healthy"



























def get_voice_data(_path):
    # select .wav files only
    wav_files = glob.glob(_path + "/*.wav")
    _type = _path.split("/")[-1] # identify type: my_data, healthy, functional etc...
    
    # list to hold voice data before turning it into a dataframe
    data = []
    
    # for each audio file,
    for wav_file in tqdm(wav_files): # tqdm shows the progress bar
        sound = parselmouth.Sound(wav_file) # sound object from wav file
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        # name analysis
        name = os.path.basename(wav_file).split(".")[0]  

        ## tone
        tone = ""
        if "l" in name:
            tone = "l"
        elif "n" in name:
            tone = "n"
        elif "h" in name:
            tone = "h"

        ## syllable
        syllab = ""
        if "a" in name:
            syllab = "a"
        elif "i" in name:
            syllab = "i"
        elif "u" in name:
            syllab = "u"

        # jitter
        jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100

        # shimmer
        shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        # append a bit before adding mfcc
        data_row = [name, _type, tone, syllab, jitter, shimmer, hnr]

        # MFCC, d1, d2
        samplerate, wav_data = wavfile.read(wav_file)
        mfccs = speechpy.feature.mfcc(wav_data, samplerate, num_cepstral = 12)
        mfccs = mfccs.T # transform to handle wav_data easily 
        derivatives = speechpy.feature.extract_derivative_feature(mfccs) # this now looks like: [c#][frame#][[mfcc, d1, d2]]

        mfcc_list = []
        mfcc_d1 = []
        mfcc_d2 = []

        # for each coefficient,
        for i in range(0, len(derivatives)):
            mfcc_vars = derivatives[i].T # mfcc, d1, d2

            # take the average across the entire time frame
            mfcc = statistics.mean(mfcc_vars[0])
            d1 = statistics.mean(mfcc_vars[1])
            d2 = statistics.mean(mfcc_vars[2])

            # append to the list
            mfcc_list.append(mfcc)
            mfcc_d1.append(d1)
            mfcc_d2.append(d2)

        data_row = data_row + mfcc_list + mfcc_d1 + mfcc_d2

        # append to data
        data.append(data_row)
        
        return data


# set up dataframe info
columns = ["Name", "Type", "Tone", "Syllab", "Jitter", "Shimmer", "HNR"]
for i in range(0,12):
    columns.append("MFCC-"+str(i))
for i in range(0,12):
    columns.append("MFCC-"+str(i)+"_d1")
for i in range(0,12):
    columns.append("MFCC-"+str(i)+"_d2")



# create dataframe
data = get_voice_data(_path)
df = pd.DataFrame(data, columns=columns)
df

































