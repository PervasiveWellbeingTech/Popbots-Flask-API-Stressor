#import tensorflow as tf
#from tensorflow.keras.models import save_model, load_model
#from tensorflow.keras.preprocessing.text import Tokenizer
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


#with open(RPATH+'tokenizer.pickle', 'rb') as handle:
    #tokenizer = pickle.load(handle) # loading the tokenizer

def data_prep(stressor,tokenizer,maxlen):
    stressor = str(stressor) #making sure that it is really of type string
    stressor = stressor.lower()
    stressor = ml.clean_text(stressor)
    if(tokenizer != None):
        stressor = tokenizer.texts_to_sequences(X_test)
        stressor = pad_sequences(stressor, maxlen=maxlen)
        
    return stressor


def lstm_predict(stressor,modelname = 'LSTM_simple_classifier.h5'):
    stressor = data_prep(stressor,tokenizer,maxlen) # tokenizing and cleaning the input

    model  = load_model(RPATH+model_name) # loading the keras model
    pred = model.predict(stressor)
    max_index  = pred.argmax(axis=1) 
    probability_max = pred[max_index]
    cat_name = category_list[pred] # return the category name form the argmax pred
    distance = probability_max - second_largest(pred,max_index) # return the distance between the max pred and the item under it
    return {"category":cat_name,"probability": probability_max,"confidence":distance}
def second_largest(l,maxIndex):
    max_2 = 0
    for ind,ele in enumerate(l):
        if (ind != maxIndex):
            if (max_2 < ele):
                max_2 = ele
    return max_2


