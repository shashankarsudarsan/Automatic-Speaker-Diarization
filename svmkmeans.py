# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:50:16 2017

@author: Shashankar
"""

import numpy as np
import librosa
import librosa.display
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import wave
import librosa
import librosa.display

import math
#KL distance function
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

#My Autocorrelation function because numpy's correlate sucks
def xcorr_max(x):
    x = np.array(x)
    result = []
    for i in range(1,len(x)+1):
        result.append((np.sum(x[:i]*x[-i:]))/(np.sum(x[:len(x)]*x[-len(x):])))
    return result

#Sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

wav_file = wave.open("SBC057.wav",mode='rb')
#Extract Raw Audio from Wav File
signal = wav_file.readframes(-1)
signal = np.fromstring(signal, 'Int16')

#Split the data into channels 
temp1 = [signal[0::2]]
temp2 = [signal[1::2]]
channels = np.concatenate((temp1,temp2),0)
del temp1, temp2, signal
#Sampling frequency, length, window size
fs = wav_file.getframerate()
length = wav_file.getnframes()
window = 1024

#Read the transcript file
transcript = open("SBC057.txt", "r")
start=[]
end=[]
speakers =[]
for columns in ( raw.strip().split() for raw in transcript ):
    start.append(columns[0])
    end.append(columns[1])
    speakers.append(columns[2])
start = list(map(float, start))
end = list(map(float, end))
start = np.array(start)
end = np.array(end)
start_frames = np.floor(start*fs/window)
end_frames = np.floor(end*fs/window)
frames = math.ceil(len(channels[0])/window)-2 #To avoid overflow
no_frames = np.arange(frames)

#Determining labels for the neural network
cleaned_speakers = []
iteration = 0
for elem in speakers:
    if elem == 'NICK:' or elem == 'X:' or elem == 'BENTON:' or elem == 'JED:' or elem == 'BILL:' or elem == 'DARREN:' or elem == 'MARTIN:':
        cleaned_speakers.append(elem)
    else:
        cleaned_speakers.append(cleaned_speakers[iteration-1])
    iteration = iteration + 1        
labels=[]    
j=0;
for i in range(int(start_frames[-1])):
    if(i<start_frames[j+1]):
        labels.append(cleaned_speakers[j])
    else:
        j+=1
#Cheating remaining labels
for i in range(frames-np.shape(labels)[0]):
    labels.append(cleaned_speakers[-1])
#Convert labels to one hot vectors
num_label = []
for elem in labels:
    if(elem == 'BENTON:'):
        num_label.append(0)
    elif(elem == 'BILL:'):
        num_label.append(1)    
    elif(elem == 'DARREN:'):
        num_label.append(2)
    elif(elem == 'JED:'):
        num_label.append(3)    
    elif(elem == 'MARTIN:'):
        num_label.append(4)
    elif(elem == 'NICK:'):
        num_label.append(5)
    else:
        num_label.append(6)        
num_label = np.array(num_label)                
one_hot_labels = np.zeros((frames,len(set(labels))))
one_hot_labels[np.arange(frames), num_label] = 1
one_hot_labels = one_hot_labels.astype(int)

#Looping through the whole signal
dc_shift = 35000; #DC Shift
KL_Dist = []
for i in range(0,frames):
    shift_first = channels[0,i*window:i*window + window] + dc_shift
    shift_second = channels[0,i*window + window:i*window + 2*window] + dc_shift
    first_seg = shift_first/np.sum(shift_first)
    second_seg = shift_second/np.sum(shift_second)
    KL_Dist.append(KL(first_seg, second_seg)+KL(second_seg, first_seg))


#Finding indices after removing repetitions by thresholding KL distance
i=0
KL_Thresh=[]
index = []
while(i < len(KL_Dist)):
    if KL_Dist[i] >= 8.362e-4:
        KL_Thresh.append(KL_Dist[i])
        index.append(i)
        i=i+10 #Choosing number of frames to skip
    else:
        i=i+1
        
KL_Thresh = np.array(KL_Thresh)        

# Loading signal
y, sr = librosa.load("SBC057.wav")

#Initialization
win_size=1024
vec_length=int(np.size(y)/win_size)+1                 
#mfccs1=np.zeros((vec_length,26))
mfccs_f=np.zeros((vec_length,13)) 
mfccs_ft=np.zeros((vec_length,1)) 
index1=np.zeros((vec_length,1))
ct=0

np.lib.pad(y, (0,1024), 'edge')

#Calculating MFCCs(pair-wise) 

for i in range(0,np.size(y),win_size):
    mfccs11=librosa.feature.mfcc(y=y[i:1023+i], sr=sr, n_mfcc=13)
    index1[int(i/win_size)]=i
    mfccs11=np.transpose(mfccs11)
    mfccs_f[int(i/win_size)]=np.transpose(mfccs11[0][0:13])


#-----------------ignore--------------------------------------------------------------------------------
#    if(np.size(mfccs11)==26):
#        mfccs1[int(i/win_size)]=np.reshape(mfccs11,(1,np.size(mfccs11)))
#    else:
#        mfccs1[int(i/win_size)]=np.lib.pad(mfccs11, [(0,0),(0,26-np.size(mfccs11))], 'edge')
#-------------------------------------------------------------------------------------------------------        
    
#Calculating Spectral Flatness as gm/am
am=np.zeros((vec_length,1))
gm=np.zeros((vec_length,1))
spec_flatness=np.zeros((vec_length,1))
ct=0

#Calculating Loudness as a measure of bark scale


#Removing zeros for gm
for i in range(np.size(y)):
    if(y[i]==0):
        ct+=1
        y[i]=1
     
for i in range(0,np.size(y),win_size):
    am[int(i/win_size)]=(sum(y[i:i+win_size-1])-ct)/len(y[i:i+win_size-1])
    gm[int(i/win_size)]=np.prod(y[i:i+win_size-1])**(1/win_size)
    spec_flatness[int(i/win_size)]=1/am[int(i/win_size)]
    mfccs_ft[int(i/win_size)]=sum(mfccs_f[int(i/win_size)][0:13])/13
    
#Calculating Loudness as a measure of RMS Energy
loudness_rms=librosa.feature.rmse(y=y,frame_length=1024,hop_length=1024)
mfcc = mfccs_ft[0:-2,:]
spec_flat = spec_flatness[0:-2,:]
data = np.concatenate((mfcc,spec_flat),1)
 
#Kmeans clustering
kmeans_loudness = KMeans(n_clusters=7, init='k-means++', max_iter=10000, random_state=0).fit(np.transpose(loudness_rms))
kmeans_specflat = KMeans(n_clusters=7, init='k-means++', max_iter=10000, random_state=0).fit(spec_flatness)
kmeans_mfcc = KMeans(n_clusters=7, init='k-means++', max_iter=10000, random_state=0).fit(mfccs_ft)
kmeans_mfcc1 = KMeans(n_clusters=7, init='k-means++', max_iter=10000, random_state=0).fit(mfccs_f)
kmeans_ens = KMeans(n_clusters=7, init='k-means++', max_iter=10000, random_state=0).fit(data)

label_l=kmeans_loudness.labels_

label_s=kmeans_specflat.labels_

label_m=kmeans_mfcc.labels_

label_m1=kmeans_mfcc1.labels_
label_m2=np.ones((vec_length,1))

#Cluster normalization

norm_ll=int(start_frames[0]);
norm_ul=int(start_frames[0]);
for i in range(0,np.size(start_frames)-1):
    print(i)
    if(norm_ll>=int(start_frames[i+1])):
        norm_ul=int(start_frames[i+2])
        continue
    norm_ul=int(start_frames[i+1])
    counts=np.bincount(label_m1[norm_ll:norm_ul])
    label_m2[norm_ll:norm_ul]=np.argmax(counts)*label_m2[norm_ll:norm_ul]
    norm_ll=norm_ul

#SVM Classification
clasf=OneVsRestClassifier(LinearSVC(random_state=0)).fit(data[0:16000], num_label[0:16000]).predict(data[19000:19500])

kmeans_ens_acc=vec_length*100/kmeans_mfcc.inertia_
print("The accuracy is (K-means) - ")      
print(kmeans_ens_acc)

cts=0
for i in range(19000,19500):
    if(clasf[i-19000]==num_label[i]):
        cts+=1
svm_acc=cts/(19500-19000)*100
print("The accuracy is (SVM) - ")
print(svm_acc)

