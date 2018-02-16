# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:19:42 2017

@author: Shashankar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:58:17 2017

@author: Shashankar
"""

import wave
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
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

##Arun
# Loading signal
sr = fs
y = channels[0,:]
win_size = window
vec_length = int(np.size(y)/win_size)+1                
mfccs1 = np.zeros((vec_length,26))
ct=0
np.lib.pad(y, (0,1024), 'edge')
#Calculating MFCCs(pair-wise)
for i in range(0,np.size(y),win_size):
    mfccs11 = librosa.feature.mfcc(y=y[i:1023+i], sr=sr, n_mfcc=13)
    mfccs11 = np.transpose(mfccs11)
    if(np.size(mfccs11)==26):
        mfccs1[int(i/win_size)] = np.reshape(mfccs11,(1,np.size(mfccs11)))
    else:
        mfccs1[int(i/win_size)] = np.lib.pad(mfccs11, [(0,0),(0,26-np.size(mfccs11))], 'edge')
   
   
   
#Calculating Spectral Flatness as gm/am
am=np.zeros((vec_length,1))
gm=np.zeros((vec_length,1))
spec_flatness = np.zeros((vec_length,1))
ct=0
#Removing zeros for gm
for i in range(np.size(y)):
    if(y[i]==0):
        ct+=1
        y[i] = 1
    
for i in range(0,np.size(y),win_size):
    am[int(i/win_size)] = (sum(y[i:i+win_size-1])-ct)/len(y[i:i+win_size-1])
    gm[int(i/win_size)] = np.prod(y[i:i+win_size-1])**(1/win_size)
    spec_flatness[int(i/win_size)] = 1/am[int(i/win_size)]

#Calculating Loudness as a measure of RMS Energy
loudness_rms=librosa.feature.rmse(y=y,frame_length=1024,hop_length=1024)
loudness_rms = np.transpose(loudness_rms)

#Clipping mfccs1 and spec_flat to size of 33850
mfcc = mfccs1[0:-2,:]
spec_flat = spec_flatness[0:-2,:]
loudness = loudness_rms[0:-2,:]
data = np.concatenate((mfcc,spec_flat),1)
data = np.concatenate((data,data,data),0)
one_hot_labels = np.concatenate((one_hot_labels,one_hot_labels,one_hot_labels),0)
frames = 3*frames

#Neural Network 
#Input
no_of_inp = data.shape[1]
no_of_classes = one_hot_labels.shape[1]
x = tf.placeholder(tf.float32, [None, no_of_inp], name="x")
summ_x = tf.placeholder(tf.float32, [None, no_of_inp], name="summ_x")
#Weights #Adding 3 hidden layers
#Hidden Layer 1
no_of_hidden_1 = 20
W_h_1 = tf.Variable(tf.zeros([no_of_inp, no_of_hidden_1]), name="W_h_1")
b_h_1 = tf.Variable(tf.zeros([no_of_hidden_1]), name="b_h_1")
#Hidden Layer 2
no_of_hidden_2 = 30
W_h_2 = tf.Variable(tf.zeros([no_of_hidden_1, no_of_hidden_2]), name="W_h_2")
b_h_2 = tf.Variable(tf.zeros([no_of_hidden_2]), name="b_h_2")
#Hidden Layer 1
no_of_hidden_3 = 20
W_h_3 = tf.Variable(tf.zeros([no_of_hidden_2, no_of_hidden_3]), name="W_h_3")
b_h_3 = tf.Variable(tf.zeros([no_of_hidden_3]), name="b_h_3")
#Output Layer 
W_o = tf.Variable(tf.zeros([no_of_hidden_3, no_of_classes]), name="W_o")
b_o = tf.Variable(tf.zeros([no_of_classes]), name="b_o")
#Model
split = 0.8
train_data = data[0:int(split * frames), :]
test_data =  data[int(split * frames):frames, :]
train_labels = one_hot_labels[0:int(split * frames), :]
test_labels = one_hot_labels[int(split * frames):frames, :]

h_1_op = tf.nn.softmax(tf.matmul(x, W_h_1) + b_h_1)
h_2_op = tf.nn.softmax(tf.matmul(h_1_op, W_h_2) + b_h_2)
h_3_op = tf.nn.softmax(tf.matmul(h_2_op, W_h_3) + b_h_3)
y = tf.nn.softmax(tf.matmul(h_3_op, W_o) + b_o)

y_ = tf.placeholder(tf.float32, [None, no_of_classes], name="labels")
summ_y = tf.placeholder(tf.float32, [None, no_of_classes], name="y_summ")
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#Evaluation
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)    
#Summaries
merged_summary = tf.summary.merge_all()
#Uncomment to write summaries
#writer = tf.summary.FileWriter("log/directory")
#writer.add_graph(sess.graph)
batch = 100
idx = 0
epochs = 1000
acc=[]
for i in range(epochs):
  batch_xs = train_data[idx * batch:idx * batch + batch,:]
  batch_ys = train_labels[idx * batch:idx * batch + batch,:]
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  acc.append(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
  if((idx+1) * batch + batch >= frames*split): #use only 80% for training
      idx=0
  else:
      idx+=1            
acc = np.array(acc)  
print("The accuracy is (DNN)- ")
print(sess.run(accuracy, feed_dict={x: test_data, y_: test_labels}))




