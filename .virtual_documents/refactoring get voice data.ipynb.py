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
# select .wav files only
wav_files = glob.glob(_path + "/*.wav")





_type = _path.split("/")[-1] # identify type: my_data, healthy, functional etc...


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



len(data)


data[0]


columns = ["Name", "Type", "Tone", "Syllab", "Jitter", "Shimmer", "HNR"]

for i in range(0,12):
    columns.append("MFCC-"+str(i))
for i in range(0,12):
    columns.append("MFCC-"+str(i)+"_d1")
for i in range(0,12):
    columns.append("MFCC-"+str(i)+"_d2")
columns





# # create dataframe
# df = pd.DataFrame(data, columns)
df = pd.DataFrame(data, columns=columns)
df


soundfile = _path+"/1-a_h.wav"


# MFCC, d1, d2
samplerate, data = wavfile.read(soundfile)
mfcc = speechpy.feature.mfcc(data, samplerate, num_cepstral = 12)
mfcc = mfcc.T # transform to handle data easily
derivatives = speechpy.feature.extract_derivative_feature(mfcc)


mfcc





derivatives


len(derivatives)





# mfcc-0 list
mfcc0_list = derivatives[0].T[0] # mfcc, d1, d2
mfcc0_list


len(mfcc0_list)


mfcc = statistics.mean(mfcc0_list)
mfcc








# [c#][frame#][[mfcc, d1, d2]]

for j in range(len(derivatives[0])):
    # get average of mfcc-0
    statistics.mean(derivatives[0][j][0])
    


# Process wav files to get Jitter, Shimmer, HNR, and MFCC and its derivatives

def get_voice_data(_path): 
    
    # initial vars
    
    n = 0
    d1 = 0
    d2 = 0
    mfcc_n = {}
    mfcc_d1 = {}
    mfcc_d2 = {}

    # create empty dataframe - [name, type, tone, syllab, jitter, shimmer, hnr, mfcc, mfcc_d1, mfcc_d2]

    df = pd.DataFrame({"Name":pd.Series(n_list),
                        "Type": np.nan,
                        "Tone": pd.Series(tone_list),
                        "Syllab": pd.Series(syllab_list),
                           "Jitter":pd.Series(j_list),
                           "Shimmer":pd.Series(s_list),
                           "HNR":pd.Series(h_list)})
    df["Type"]= _path.split("/")[-1] # identify type: my_data, healthy, functional etc...
    new_df = pd.concat([df, mfcc_n_df, mfcc_d1_df, mfcc_d2_df], axis=1, sort=False)

    
    # select .wav files only
    wav_files = glob.glob(_path + "/*.wav")
    
    
    # for wav_file in wav_files:
    for wav_file in tqdm(wav_files): # tqdm shows the progress bar
        sound = parselmouth.Sound(wav_file) # sound object from wav file
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        # name analysis
        name = os.path.basename(wav_file).split(".")[0]  
        
        ## tone
        if "l" in name:
            tone_list.append("l")
        elif "n" in name:
            tone_list.append("n")
        elif "h" in name:
            tone_list.append("h")

        ## syllable
        if "a" in name:
            syllab_list.append("a")
        elif "i" in name:
            syllab_list.append("i")
        elif "u" in name:
            syllab_list.append("u")
        # jitter
        jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100

        # shimmer
        shimmer_local = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        
        # Append to numpy array
        n_list.append(name)
        j_list.append(jitter_local)
        s_list.append(shimmer_local)
        h_list.append(hnr)

        # MFCC - parselmouth (PRAAT)
#         mfcc_object = sound.to_mfcc(number_of_coefficients=13)
#         mfcc_arr = mfcc_object.to_array()
#         mfcc_dic = {}
#         for i in range(0,len(mfcc_arr)):
#             mfcc_dic["MFCC-"+str(i)] = [statistics.mean(mfcc_arr[i])]
#         mfcc_df = pd.DataFrame.from_dict(mfcc_dic)
        
        
        # MFCC, d1, d2
        samplerate, data = wavfile.read(wav_file)
        mfcc = speechpy.feature.mfcc(data, samplerate, num_cepstral = 12)
        mfcc = mfcc.T # transform to handle data easily
        derivatives = speechpy.feature.extract_derivative_feature(mfcc)


        for i in range(0,len(derivatives)):
            ders = derivatives[i].T # transform to handle data easily
            n = [statistics.mean(ders[0])]
            d1 = [statistics.mean(ders[1])]
            d2 = [statistics.mean(ders[2])]
            mfcc_n["MFCC-"+str(i)] = n
            mfcc_d1["MFCC-"+str(i)+"_d1"] = d1
            mfcc_d2["MFCC-"+str(i)+"_d2"] = d2
            
            mfcc_n_df = pd.DataFrame.from_dict(mfcc_n)
            mfcc_d1_df = pd.DataFrame.from_dict(mfcc_d1)
            mfcc_d2_df = pd.DataFrame.from_dict(mfcc_d2)


    
    return new_df

