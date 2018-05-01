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

#plot the features
#for j in range(1,jetInfos_array.shape[1]):
#    jetInfosQCD=[]
#    jetInfosTop=[]
#    jetInfosHiggs=[]
#    jetInfosDY=[]
#    jetInfosW=[]

#    for i in range(0,len(Labels_array)):
#        if Labels_array[i]==0.0:
#            jetInfosQCD.append(jetInfos_array[i,j])
#        elif Labels_array[i]==1.0:
#            jetInfosTop.append(jetInfos_array[i,j])
#        elif Labels_array[i]==2.0:
#            jetInfosDY.append(jetInfos_array[i,j])
#        elif Labels_array[i]==3.0:
#            jetInfosW.append(jetInfos_array[i,j])
#        elif Labels_array[i]==4.0:
#            jetInfosHiggs.append(jetInfos_array[i,j])

#    f = plt.figure(j)
#    plt.hist(jetInfosQCD,label='QCD',histtype="step",normed=True)
#    plt.hist(jetInfosTop,label='Top',histtype="step",normed=True)
#    plt.hist(jetInfosDY,label='DY',histtype="step",normed=True)
#    plt.hist(jetInfosW,label='W',histtype="step",normed=True)
#    plt.hist(jetInfosHiggs,label='Higgs',histtype="step",normed=True)
#    plt.xlabel('Features')
#    plt.ylabel('Entries')
#    plt.legend(loc='upper right')
#    plt.title('First feature test')
#    if j<6:
#         plt.axis([0, 1, 0.0001, 99])
#    elif (j==6 or j==7 or j==10 or j==11 or j==12 or j==13 or j==14):
#         plt.axis([10, 1000, 0.0001, 99])
#    else:
#         plt.axis([-5, 5, 0.0001, 99])
#    plt.grid(True)

#    plt.yscale('log')
#    f.show()
#    f.savefig('feat'+str(j)+'.png')

for f in range(jetInfos_array.shape[1]):
    mean = np.mean(jetInfos_array[:,f])
    std = np.std(jetInfos_array[:,f])
    jetInfos_array[:,f] = (jetInfos_array[:,f] - mean)/std

#multi_labels = keras.utils.to_categorical(Labels_array, num_classes=2)

# let's try a deep neural network with the built inputs
#First part of the NN
model = Sequential()

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further

model.add(Dense(12, input_dim=13, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(10, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
#model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#model.add(Dense(800,W_regularizer=l2(0.5))) #this is the other regularizator, other than dropout

#In principle, you can give the weights to the layers by hand and require that they are not weighted
#drop-out?
#model.add(Dropout(0.01)) # dropout layer #it does not bring anything at the end of course!!

# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model (this is the training!!!)
history = model.fit(jetInfos_array, Labels_array, epochs=100, batch_size=256)

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
plot_model(model, to_file='model.png')

#printing the model summary
print(model.summary())

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

#I try to get the activation functions from the deep learning curve

# we build a new model with the activations of the old model
# this model is truncated after the first layer

#print activations
#plt.hist(activations)
#plt.title("Deep Neural Network output in test sample")
#plt.xlabel("Value")
#plt.ylabel("Number of events")
#plt.show()

# evaluate the model (this is the test!!!)
#score = model.evaluate(X_test, Y_test, verbose=0)
#scores_train = model.evaluate(X_train, Y_train,verbose=True)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))

#scores_test = model.evaluate(X_test, Y_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

JetsInfosTest=[]
LabelsTest=[]

with open('TestEventsTwoLabels.txt') as fin:
#with open('QCD-Test.txt') as fin:
    for line in fin:
        items = line.split()
        LabelsTest.append(items[0])
        JetsInfosTest.append(items[1:9]+items[10:15])

#Now the jet information
jetInfosTest_array = np.asarray(JetsInfosTest)
jetInfosTest_array = jetInfosTest_array.astype(np.float)

for f in range(jetInfosTest_array.shape[1]):
    mean = np.mean(jetInfosTest_array[:,f])
    std = np.std(jetInfosTest_array[:,f])
    jetInfosTest_array[:,f] = (jetInfosTest_array[:,f] - mean)/std

LabelsTest_array = np.asarray(LabelsTest)
LabelsTest_array = LabelsTest_array.astype(np.float)

#multi_labels_test = keras.utils.to_categorical(LabelsTest_array, num_classes=2)

predictions = model.predict(jetInfosTest_array)
predictions_array = np.asarray(predictions)
predictions_array = predictions_array.astype(np.float)

print predictions

scores_train = model.evaluate(jetInfos_array,Labels_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))

scores_train = model.evaluate(jetInfosTest_array,LabelsTest_array,verbose=True)
print("TEST \n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))

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

for f in range(jetInfosTest_array.shape[0]):
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

#including a possible Linear Discriminant analysis and a boosted decision tree analysis
clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(jetInfos_array, Labels_array)
print(clf.predict(jetInfosTest_array))

#bdt = RandomForestClassifier(n_estimators=10)
#bdt.fit(jetInfos_array, Labels_array)
#print(bdt.predict(jetInfosTest_array))
