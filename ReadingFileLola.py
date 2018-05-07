import pandas
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D
from keras import regularizers
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Dropout
from keras.models import model_from_yaml
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.utils.np_utils import to_categorical

#Lola and Cola layers
from LorentzLayer.lola import LoLa
#or
import sys
sys.path.append("../LorentzLayer")

from cola import CoLa
from lola import LoLa
from sola import SoLa

input_filename = "train.h5"
store = pandas.HDFStore(input_filename)

# Read the first 10 events
#df = store.select("table",stop=10)

# Read all events
n_events=10000
df = store.select("table",stop=n_events)

#print df #print all components of the first 10 events

#How to access the different components
pX = df['PX_0'] #component x component of transverse momentum
#print pX

#Here it is to define the name of the columns
n_cands=40

#FourVectors  = ["E_{0}".format(i_cand) for i_cand in range(n_cands)] 
#FourVectors += ["PX_{0}".format(i_cand) for i_cand in range(n_cands)]
#FourVectors += ["PY_{0}".format(i_cand) for i_cand in range(n_cands)]
#FourVectors += ["PZ_{0}".format(i_cand) for i_cand in range(n_cands)]

#print FourVectors
#for j in range(start,end):
#components=['E_'+str(component)+str(,)'PX_'+str(component)+str(,)'PY_'+str(component)+str(,)'PZ_'+str(component) for component in range(n_cands)]
#print components	

#FourVectors0 = df[['E_0','PX_0','PY_0','PZ_0']]
#FourVectors1 = df[['E_1','PX_1','PY_1','PZ_1']]
#print FourVectors0
#result = pandas.concat([FourVectors0, FourVectors1], axis=1)
#resultarray = numpy.array(result)
#b = resultarray.reshape(10,2,4)
#print b.shape
##Now you have to transpose it
#b = numpy.transpose(b,(10,4,2))
#print resultarray.shape
#print b

FourVectorsSD = {}
FourVectors = pandas.DataFrame()

for i in range(0,n_cands):
	component = i
	FourVectorsSD[i] = df[['E_'+str(component),'PX_'+str(component),'PY_'+str(component),'PZ_'+str(component)]] #four vector components
	
        FourVectors = pandas.concat([FourVectors,FourVectorsSD[i]],axis=1)

data =  FourVectors.as_matrix()
FourVectorsForTrain = data.reshape((n_events,4,n_cands),order="F")
#FourVectorsForTrain = numpy.expand_dims(df[FourVectors],axis=-1).reshape(-1,4,n_cands)

Labels = df['is_signal_new'] #Labels top/QCD

print FourVectorsForTrain.shape

#FourVectorsArray = numpy.array(FourVectors)
#FourVectorsForTrain = FourVectorsArray.reshape(n_events,4,n_cands)

input_filenameval = "val.h5"
storeval = pandas.HDFStore(input_filenameval)
dfval = storeval.select("table",stop=n_events)

FourVectorsSD = {}
FourVectors = pandas.DataFrame()

for i in range(0,n_cands):
	component = i
	FourVectorsSD[i] = dfval[['E_'+str(component),'PX_'+str(component),'PY_'+str(component),'PZ_'+str(component)]] #four vector components
	
        FourVectors = pandas.concat([FourVectors,FourVectorsSD[i]],axis=1)

data =  FourVectors.as_matrix()
FourVectorsForVal = data.reshape((n_events,4,n_cands),order="F")

LabelsVal = dfval['is_signal_new'] #Labels top/QCD
LabelsVal_binary = to_categorical(LabelsVal)

#print FourVectors.shape

#From that list, I want to select only the energy now
#print FourVectors['E_0']

#From that list, I want to select only the first event
#print FourVectors.loc[375,:] #the index of the first event is 375

#Now you can start to feed your neural network with the components

#Here is the LoLa model, which is inside LorentzLayer/Models.py
model = Sequential()

model.add(CoLa(input_shape = (4, 2),
               add_total   = False,
               add_eye     = True,
               debug =  False,
               n_out_particles = 5))

model.add(LoLa(
	train_metric = False,
        es  = 0,
	xs  = 0,
        ys  = 0,
	zs  = 0,
        cs  = 0,
        vxs = 0,
	vys = 0,
        vzs = 0,
        ms  = 1,
	pts = 1,
        n_train_es  = 1,
        n_train_ms  = 0,
	n_train_pts = 0,
        n_train_sum_dijs   = 2,
	n_train_min_dijs   = 2))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10, activation='softmax'))

#Let's try to include an already defined function
sys.path.append("dnn_template")

from Models import model_lola

#you have to define the parameters, as included in the model_lola function

print 'We are here..it works!'

my_params = {
    # Parameters for constituent approach                                                                                                                                                                
    "n_out_particles" : 15,
    "n_constit" : n_cands,
    "n_features" : 4,
    "n_classes" : 2,
    "lr"   : 0.00001,
}

model_predefined = model_lola(my_params)
model_predefined.summary()

#After defining the model, as usual, you can train it
model_predefined.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
# Fit the model (this is the training!)
#It wants a matrix with the four components and the particles
Labels_binary = to_categorical(Labels)

history = model_predefined.fit(FourVectorsForTrain, Labels_binary, validation_data=(FourVectorsForVal, LabelsVal_binary), epochs=10, batch_size=256)

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

#Defining the test values
input_filenametest = "test.h5"
store = pandas.HDFStore(input_filenametest)
dftest = store.select("table",stop=n_events)

FourVectorsSD = {}
FourVectors = pandas.DataFrame()

for i in range(0,n_cands):
	component = i
	FourVectorsSD[i] = dftest[['E_'+str(component),'PX_'+str(component),'PY_'+str(component),'PZ_'+str(component)]] #four vector components
	
        FourVectors = pandas.concat([FourVectors,FourVectorsSD[i]],axis=1)

data =  FourVectors.as_matrix()
FourVectorsForTest = data.reshape((n_events,4,n_cands),order="F")

LabelsTest = dftest['is_signal_new'] #Labels top/QCD

predictions = model_predefined.predict(FourVectorsForTest)

print ' '

scores_train = model_predefined.evaluate(FourVectorsForTrain,Labels_binary,verbose=True)
print("TRAIN TOP score \n%s: %.2f%%" % (model_predefined.metrics_names[1], scores_train[1]*100))

print ' '

LabelsTest_binary = to_categorical(LabelsTest)

scores_test = model_predefined.evaluate(FourVectorsForTest,LabelsTest_binary,verbose=True)
print("TEST TOP score \n%s: %.2f%%" % (model_predefined.metrics_names[1], scores_test[1]*100))

print ' '

fpr, tpr, _ = roc_curve(LabelsTest_binary[:,1], predictions[:,1])
roc_auc_TOP = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_TOP)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
print('AUC: %f' % roc_auc_TOP)

plt.show()

scores_train = model_predefined.evaluate(FourVectorsForTrain,Labels_binary,verbose=True)
print("TRAIN QCD score \n%s: %.2f%%" % (model_predefined.metrics_names[1], scores_train[0]*100))

print ' '

scores_test = model_predefined.evaluate(FourVectorsForTest,LabelsTest_binary,verbose=True)
print("TEST QCD score \n%s: %.2f%%" % (model_predefined.metrics_names[1], scores_test[0]*100))

print ' '

fpr, tpr, _ = roc_curve(LabelsTest_binary[:,0], predictions[:,0])
roc_auc_QCD = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_QCD)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
print('AUC: %f' % roc_auc_QCD)

plt.show()

predictions_Bkg=[]
predictions_Sig=[]

for f in range(LabelsTest_binary.shape[0]):
    if(LabelsTest_binary[f,0]==0):
        predictions_Bkg.append(predictions[f,0])
    elif(LabelsTest_binary[f,0]==1):
        predictions_Sig.append(predictions[f,0])

predictions_Bkg_array = numpy.asarray(predictions_Bkg)
predictions_Bkg_array = predictions_Bkg_array.astype(numpy.float)

predictions_Sig_array = numpy.asarray(predictions_Sig)
predictions_Sig_array = predictions_Sig_array.astype(numpy.float)

f = plt.figure()
plt.hist(predictions_Bkg_array,label='QCD',normed=True,alpha = 0.5)
plt.hist(predictions_Sig_array,label='TOP',normed=True,alpha = 0.5)
plt.legend(loc='upper right')
plt.title("DNN Output")
plt.xlabel("Value")
plt.ylabel("Number Of Events")
f.show()
f.savefig('Output.png')


