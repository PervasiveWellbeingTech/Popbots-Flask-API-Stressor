
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import numpy as np 
import pickle
import mlutils as ml

"""
Const variables needed in the process
"""
RPATH = './ressources/'
embedding_matrix = np.loadtxt(RPATH+'embedding_matrix.txt') # loading the embedding matrix trained obtained during training
category_list = ['Confidence Issue', 'Everyday Decision Making', 'Exhaustion/Fatigue',
       'Family Issues', 'Financial Problem', 'Health or Physical Pain',
       'Other', 'Personal/Social Issues', 'Travel/Holiday Stress',
       'Work/School Productivity']

model  = load_model(RPATH+'LSTM_simple_classifier.h5') # loading the keras model


with open(RPATH+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle) # loading the tokenizer

def data_prep(stressor,tokenizer,maxlen):
    stressor = str(stressor) #making sure that it is really of type string
    stressor = stressor.lower()
    stressor = ml.clean_text(stressor)
    #print("clearned stressor"+str(stressor))
    if(tokenizer != None):
        stressor = tokenizer.texts_to_sequences([stressor])
        # print("token stressor"+str(stressor))
        stressor = pad_sequences(stressor, maxlen=maxlen)
        #print("pad stressor"+str(stressor))
    return stressor


def lstm_predict(stressor,model_name = 'LSTM_simple_classifier.h5'):
    stressor = data_prep(stressor,tokenizer,maxlen=100) # tokenizing and cleaning the input
    print(stressor)
    #with tf.device('/cpu:0'):
    model  = load_model(RPATH+model_name) # loading the keras model
    #Do CPU stuff here
    pred = model.predict(stressor)
    print(pred)
    max_index  = pred.argmax(axis=1)
    print("index is "+str(pred))
    probability_max = pred[0][max_index]
    cat_name = category_list[int(max_index)] # return the category name form the argmax pred
    distance = probability_max - second_largest(pred[0],max_index) # return the distance between the max pred and the item under it
    return {"category":str(cat_name),"probability": str(probability_max),"confidence":str(distance)}
def second_largest(l,maxIndex):
    max_2 = 0
    for ind,ele in enumerate(l):
        if (ind != maxIndex):
            if (max_2 < ele):
                max_2 = ele
    return max_2


