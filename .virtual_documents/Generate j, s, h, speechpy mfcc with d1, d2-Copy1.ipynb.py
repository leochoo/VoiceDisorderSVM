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



# Process wav files to get Jitter, Shimmer, HNR, and MFCC

def get_voice_data(_path):
    # select .wav files only
    wav_files = glob.glob(_path + "/*.wav")

    n_list = []
    tone_list = []
    syllab_list = []

    j_list = []
    s_list = []
    h_list = []
    
    n = 0
    d1 = 0
    d2 = 0
    mfcc_n = {}
    mfcc_d1 = {}
    mfcc_d2 = {}

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


    # create dataframe
    df = pd.DataFrame({"Name":pd.Series(n_list),
                        "Type": np.nan,
                        "Tone": pd.Series(tone_list),
                        "Syllab": pd.Series(syllab_list),
                           "Jitter":pd.Series(j_list),
                           "Shimmer":pd.Series(s_list),
                           "HNR":pd.Series(h_list)})
    df["Type"]= _path.split("/")[-1] # identify type: my_data, healthy, functional etc...
    new_df = pd.concat([df, mfcc_n_df, mfcc_d1_df, mfcc_d2_df], axis=1, sort=False)

    return new_df



def generate_jshmfcc(dataset_type, dataset_path):
    healthy_df = get_voice_data(dataset_path + "/healthy")
    functional_df = get_voice_data(dataset_path + "/pathological/functional")
    hyperfunctional_df = get_voice_data(dataset_path + "/pathological/hyperfunctional")
    organic_df = get_voice_data(dataset_path + "/pathological/organic")
    psychogenic_df = get_voice_data(dataset_path + "/pathological/psychogenic")

    # Combine the results into one dataframe
    frames = [healthy_df, functional_df, hyperfunctional_df, organic_df, psychogenic_df]
    combined_df = pd.concat(frames)
    combined_df = combined_df.dropna()
    return combined_df



# filepath for the test and train datasets
test_path = "/Users/leochoo/dev/VoiceDisorderSVM/data/SVD/test_audio"
train_path = "/Users/leochoo/dev/VoiceDisorderSVM/data/SVD/train_audio"


# generate voice report for test dataset
test_report = generate_jshmfcc("test", test_path)
test_report


# # generate voice report for train dataset
# train_report = generate_jshmfcc("train", train_path)
# train_report


# Save the outputs to the processed data directory
test_report.to_csv ("./data/processed/test_SVD_j_s_hnr_mfcc_with_d1d2.csv", index = False, header=True)
print("Test data exported")
# train_report.to_csv ("./data/processed/train_SVD_j_s_hnr_mfcc_withd1d2.csv", index = False, header=True)
# print("Train data exported")





# 20201105 
# so i recognized the problem with mfcc calculation so I'm re-doing it correctly.

# 1105 09:02 now generating new dataset with the correct average mfcc value. no d1 d2 included here.









