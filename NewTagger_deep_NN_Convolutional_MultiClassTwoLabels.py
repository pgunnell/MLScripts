# Create your first MLP in Keras
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
import csv
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset

Labels=[]
ChargedPt=[]
NeutralPt=[]
Header_2=[]
Header_3=[]
Header_4=[]
JetsInfos=[]
ConstInfos=[]

#weight initialization

with open('TrainEventsTwoLabels.txt') as fin:
    for line in fin:
        items = line.split()
        Labels.append(items[0])
        JetsInfos.append(items[1:9]+items[10:15])
        #print items[1:19]
        ChargedPt.append(items[20:84])
        NeutralPt.append(items[85:149])
        Header_2.append(items[19]) 
        Header_3.append(items[85])
        Header_4.append(items[150])
###Old way to load txt files -> does not work for files with different lengths (but works for string and float in the file
#datasetTrain = numpy.genfromtxt("DY.txt", delimiter=" ", dtype=None)
# split into input (X) and output (Y) variables
#X_train = datasetTrain[:,2:65]
#Y_train = datasetTrain[:,0]

#The output vector is a list of string values 
#For using that in histos or machine learning techniques, one has to convert the lists into arrays, reshape the jet images in matrix (8x8)  and convert the strings to floats
#This is what is done in the following for all our variables
chargedpt_array_image_one = np.asarray(ChargedPt[1])
chargedpt_array_image = [i.split(' ', 1)[0] for i in chargedpt_array_image_one]
ChargedPt_Image = np.reshape(chargedpt_array_image,(8,8))
ChargedPt_Image = ChargedPt_Image.astype(np.float)
#print ChargedPt_Image

neutralpt_array_image_one = np.asarray(NeutralPt[1])
neutralpt_array_image = [i.split(' ', 1)[0] for i in neutralpt_array_image_one]
NeutralPt_Image = np.reshape(neutralpt_array_image,(8,8))
NeutralPt_Image = NeutralPt_Image.astype(np.float)

#plt.imshow(NeutralPt_Image)
#plt.title("Jet image around the axis")
#plt.xlabel("Pseudorapidity")
#plt.ylabel("Phi angle")

#plt.colorbar()
#plt.show()
#savefig('DrellYan.png')

#Now you try to set the arrays for the deep learning analysis -> at the end, you will have 8x8 matrices for the images
chargedpt_array = np.asarray(ChargedPt)
chargedpt_array = chargedpt_array.astype(np.float)
shapeCharged = chargedpt_array.shape[0]
chargedpt_array = chargedpt_array.reshape(shapeCharged,8,8,1)

neutralpt_array = np.asarray(NeutralPt)
neutralpt_array = neutralpt_array.astype(np.float)
shapeNeutral = neutralpt_array.shape[0]
neutralpt_array = neutralpt_array.reshape(shapeNeutral,8,8,1)

#Now the jet information
jetInfos_array = np.asarray(JetsInfos)
jetInfos_array = jetInfos_array.astype(np.float)

#Now the labels
Labels_array = np.asarray(Labels)
Labels_array = Labels_array.astype(np.float)

for f in range(chargedpt_array.shape[1]):
    for g in range(chargedpt_array.shape[2]):
        mean = np.mean(chargedpt_array[:,f,g,:])
        std = np.std(chargedpt_array[:,f,g,:])
        if(std!=0):
            chargedpt_array[:,f,g,:] = (chargedpt_array[:,f,g,:] - mean)/std

for f in range(neutralpt_array.shape[1]):
    for g in range(neutralpt_array.shape[2]):
        mean = np.mean(neutralpt_array[:,f,g,:])
        std = np.std(neutralpt_array[:,f,g,:])
        if(std!=0):
            neutralpt_array[:,f,g,:] = (neutralpt_array[:,f,g,:] - mean)/std

Images_array = np.concatenate((chargedpt_array,neutralpt_array),axis=3)
input_shape = (8, 8, 2)

#First part of the NN
model_CNN = Sequential()
model_CNN.add(Conv2D(128, kernel_size=(2,2), padding='same',input_shape=input_shape,activation='relu'))
#model_CNN.add(Conv2D(16,(2,2),activation='relu'))
model_CNN.add(Dropout(0.1))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.1))
#model_CNN.add(Conv2D(16,(3,3),activation='relu'))
#model_CNN.add(Dropout(0.25))
model_CNN.add(Flatten())
#model_CNN.add(Dense(128, activation='relu'))
#model_CNN.add(Dense(128, activation='relu'))
model_CNN.add(Dense(16, activation='sigmoid'))
model_CNN.add(Dense(1, activation='sigmoid'))

model_CNN.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model_CNN.fit(Images_array, Labels_array,batch_size=256,epochs=80,verbose=1)

#plotting the structure of the NN
from keras.utils import plot_model
plot_model(model_CNN, to_file='model_CNN.png')

#printing the model summary
print(model_CNN.summary())

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

LabelsTest=[]
ChargedPtTest=[]
NeutralPtTest=[]
Header_2Test=[]
Header_3Test=[]
Header_4Test=[]
JetsInfosTest=[]
ConstInfosTest=[]
ConstInfos_TotalTest=[]

#weight initialization
with open('TestEventsTwoLabels.txt') as fin:
    for line in fin:
        items = line.split()
        LabelsTest.append(items[0])
        ChargedPtTest.append(items[20:84])
        NeutralPtTest.append(items[85:149])

LabelsTest_array = np.asarray(LabelsTest)
LabelsTest_array = LabelsTest_array.astype(np.float)

chargedptTest_array_image_one = np.asarray(ChargedPtTest[1])
chargedptTest_array_image = [i.split(' ', 1)[0] for i in chargedptTest_array_image_one]
ChargedPtTest_Image = np.reshape(chargedptTest_array_image,(8,8))
ChargedPtTest_Image = ChargedPtTest_Image.astype(np.float)

neutralptTest_array_image_one = np.asarray(NeutralPtTest[1])
neutralptTest_array_image = [i.split(' ', 1)[0] for i in neutralptTest_array_image_one]
NeutralPtTest_Image = np.reshape(neutralptTest_array_image,(8,8))
NeutralPtTest_Image = NeutralPtTest_Image.astype(np.float)

#Now you try to set the arrays for the deep learning analysis -> at the end, you will have 8x8 matrices for the images
chargedptTest_array = np.asarray(ChargedPtTest)
chargedptTest_array = chargedptTest_array.astype(np.float)
shapeCharged = chargedptTest_array.shape[0]
chargedptTest_array = chargedptTest_array.reshape(shapeCharged,8,8,1)

neutralptTest_array = np.asarray(NeutralPtTest)
neutralptTest_array = neutralptTest_array.astype(np.float)
shapeNeutral = neutralptTest_array.shape[0]
neutralptTest_array = neutralptTest_array.reshape(shapeNeutral,8,8,1)

for f in range(chargedptTest_array.shape[1]):
    for g in range(chargedptTest_array.shape[2]):
        mean = np.mean(chargedptTest_array[:,f,g,:])
        std = np.std(chargedptTest_array[:,f,g,:])
        if(std!=0):
            chargedptTest_array[:,f,g,:] = (chargedptTest_array[:,f,g,:] - mean)/std

for f in range(neutralptTest_array.shape[1]):
    for g in range(neutralptTest_array.shape[2]):
        mean = np.mean(neutralptTest_array[:,f,g,:])
        std = np.std(neutralptTest_array[:,f,g,:])
        if(std!=0):
            neutralptTest_array[:,f,g,:] = (neutralptTest_array[:,f,g,:] - mean)/std

ImagesTest_array = np.concatenate((chargedptTest_array,neutralptTest_array),axis=3)

predictions = model_CNN.predict(ImagesTest_array)
predictions_array = np.asarray(predictions)
predictions_array = predictions_array.astype(np.float)

print predictions

scores_train = model_CNN.evaluate(Images_array,Labels_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model_CNN.metrics_names[1], scores_train[1]*100))

scores_train = model_CNN.evaluate(ImagesTest_array,LabelsTest_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model_CNN.metrics_names[1], scores_train[1]*100))

#model.fit(jetInfos_array,multi_labels, validation_data=(jetInfosTest_array,multi_labels_test), epochs=150, batch_size=10) #this is a very good way to cross validate your network! Otherwise the manual k-fold

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

for f in range(LabelsTest_array.shape[0]):
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

#bdt = RandomForestClassifier(n_estimators=10)
#bdt.fit(jetInfos_array, Labels_array)
#print(bdt.predict(jetInfosTest_array))
