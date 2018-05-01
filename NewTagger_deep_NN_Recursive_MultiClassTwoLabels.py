# Create your first MLP in Keras
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
import csv
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.layers import Embedding
from keras.layers import LSTM, SimpleRNN 

from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset

# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset

Labels=[]
ChargedPt=[]
NeutralPt=[]
Header_1=[]
Header_2=[]
Header_3=[]
Header_4=[]
JetsInfos=[]
ConstInfos=[]
ConstInfos_Total=[]

with open('TrainEventsTwoLabels.txt') as fin:
    for line in fin:
        ConstInfos=np.zeros(140)
        items = line.split()
        length = 674
        if len(items)< 674:
            length=len(items)
        Labels.append(items[0])
        for j in range(534,length):
            ConstInfos[j-534]=items[j]
        ConstInfos_Total.append(ConstInfos)

#Now the constituents fix in the text files
ConstInfos_array = np.asarray(ConstInfos_Total)
ConstInfos_array = ConstInfos_array.astype(np.float)
#print ConstInfos_array.shape

for f in range(ConstInfos_array.shape[1]):
    mean = np.mean(ConstInfos_array[:,f])
    std = np.std(ConstInfos_array[:,f])
    ConstInfos_array[:,f] = (ConstInfos_array[:,f] - mean)/std

#ConstInfos_array.sort(axis=0)
ConstInfos_array=ConstInfos_array.reshape(ConstInfos_array.shape[0],20,7)

#ConstInfos_array= np.sort(ConstInfos_array.view('f8,f8,f8'), order=['f1'], axis=0).view(np.float)
#ConstInfos_array= sorted(ConstInfos_array,key=lambda ConstInfos_array_entry:ConstInfos_array_entry[1])

#Now the labels
Labels_array = np.asarray(Labels)
Labels_array = Labels_array.astype(np.float)

model_LSTM = Sequential()
model_LSTM.add(SimpleRNN(25, return_sequences=True,input_shape=(20,7), kernel_initializer='random_uniform',bias_initializer='zeros'))  # returns a sequence of vectors of dimension 32
model_LSTM.add(SimpleRNN(15, return_sequences=True,kernel_initializer='random_uniform',bias_initializer='zeros'))  # returns a sequence of vectors of dimension 150
model_LSTM.add(SimpleRNN(10, return_sequences=True))  # returns a sequence of vectors of dimension 150
model_LSTM.add(SimpleRNN(4))  # return a single vector of dimension 32
model_LSTM.add(Dense(32, activation='relu'))
model_LSTM.add(Dense(10, activation='relu'))
model_LSTM.add(Dense(1, activation='relu'))

# Compile model
model_LSTM.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# Fit the model (this is the training!!!)
history = model_LSTM.fit(ConstInfos_array,  Labels_array, epochs=100, batch_size=512)
#here starts the convolutional neural network

#optimizing your deep network is equivalent to change the parameters of the network itself: adding a new layer, change the learning rate, change the loss function, change the optimizers
#from keras.optimizers import RMSprop
#rmsprop = RMSprop(lr=0.0001) #change the learning rate
#model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])

#from keras.optimizers import SGD, RMSprop
#sgd=SGD(lr=0.1)  #gradient descent
#model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

#plotting the structure of the NN
from keras.utils import plot_model
plot_model(model_LSTM, to_file='modelLSTM.png')

#printing the model summary
print(model_LSTM.summary())

#plot the loss as a function of the epoch
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Load the test infos
LabelsTest=[]
ConstInfosTest=[]
ConstInfosTest_Total=[]

with open('TestEventsTwoLabels.txt') as fin:
    for line in fin:
        ConstInfosTest=np.zeros(140)
        items = line.split()
        length = 674
        if len(items)< 674:
            length=len(items)
        LabelsTest.append(items[0])
        for j in range(534,length):
            ConstInfosTest[j-534]=items[j]
        ConstInfosTest_Total.append(ConstInfosTest)

ConstInfosTest_array = np.asarray(ConstInfosTest_Total)
ConstInfosTest_array = ConstInfosTest_array.astype(np.float)

for f in range(ConstInfosTest_array.shape[1]):
    mean = np.mean(ConstInfosTest_array[:,f])
    std = np.std(ConstInfosTest_array[:,f])
    ConstInfosTest_array[:,f] = (ConstInfosTest_array[:,f] - mean)/std

ConstInfosTest_array=ConstInfosTest_array.reshape(ConstInfosTest_array.shape[0],20,7)

#Now the labels
LabelsTest_array = np.asarray(LabelsTest)
LabelsTest_array = LabelsTest_array.astype(np.float)

predictions = model_LSTM.predict(ConstInfosTest_array)
predictions_array = np.asarray(predictions)
predictions_array = predictions_array.astype(np.float)

print predictions

scores_train = model_LSTM.evaluate(ConstInfos_array,Labels_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model_LSTM.metrics_names[1], scores_train[1]*100))

scores_train = model_LSTM.evaluate(ConstInfosTest_array,LabelsTest_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model_LSTM.metrics_names[1], scores_train[1]*100))

fpr, tpr, _ = roc_curve(LabelsTest_array, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
print('AUC: %f' % roc_auc)

plt.show()

fileOut = open("predictions.txt","w") 
for value in range(predictions.shape[0]):
    #fileOut.write(str(predictions[value,0])+" "+str(predictions[value,1])+"\n")
    fileOut.write(str(predictions_array[value])+"\n")
fileOut.close()

predictions_Bkg=[]
predictions_Sig=[]

for f in range(ConstInfosTest_array.shape[0]):
    if(LabelsTest_array[f]==0):
        predictions_Bkg.append(predictions_array[f])
    elif(LabelsTest_array[f]==1):
        predictions_Sig.append(predictions_array[f])

predictions_Bkg_array = np.asarray(predictions_Bkg)
predictions_Bkg_array = predictions_Bkg_array.astype(np.float)

predictions_Sig_array = np.asarray(predictions_Sig)
predictions_Sig_array = predictions_Sig_array.astype(np.float)

f = plt.figure()
plt.hist(predictions_Bkg_array,label='QCD',normed=True,alpha = 0.5)
plt.hist(predictions_Sig_array,label='DY',normed=True,alpha = 0.5)
plt.legend(loc='upper right')
plt.title("DNN Output")
plt.xlabel("Value")
plt.ylabel("Number Of Events")
f.show()
f.savefig('Output.png')
