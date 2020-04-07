
import tensorflow as tf
import tensorflow_hub as hub
import requests
import numpy as np 
import pickle
import mlutils as ml

#!pip install bert-tensorflow
import bert
from bert import run_classifier
from bert import run_classifier_with_tfhub
from bert import optimization
from bert import tokenization

"""
Const variables needed in the process
"""
RPATH = './ressources/'

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
TENSOR_SERVER_URL = "http://commuter.stanford.edu:8501/v1/models/bertstressor:predict"

MAX_SEQ_LENGTH = 128

category_list = ['Emotional Turmoil', 'Work', 'School', 'Health or Physical Pain', 'Financial Problem', 'Family Issue']


# This is a path to an uncased (all lowercase) version of BERT

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer=create_tokenizer_from_hub_module()
def data_prep(stressors_list,label_list, MAX_SEQ_LENGTH, tokenizer):    
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 1) for x in stressors_list] 
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        
    return input_features

def softmax(nparray):
    return np.divide(1,np.add(1,np.exp(-np.array(nparray))))
def exp(nparray):
    return np.exp(np.array(nparray)) 
def bert_predict(stressor):
    #tokenizer=create_tokenizer_from_hub_module()
    stressors_ids = data_prep([stressor],range(len(category_list)), MAX_SEQ_LENGTH, tokenizer)
    pred = exp(get_pred_api(stressors_ids[0].input_ids))
    max_index  = np.array(pred).argmax()
    print("index is "+str(max_index))
    probability_max = pred[max_index]
    max_2,max_2_index= second_largest(pred,max_index)
    cat_name = category_list[max_index] # return the category name form the argmax pred
    distance = probability_max - max_2 # return the distance between the max pred and the item under it
    return {"probabilities":list(pred) ,"category":str(cat_name),"category2":str(category_list[max_2_index]),"probability": str(probability_max),"confidence":str(distance)}

def second_largest(l,maxIndex):
    max_2 = 0
    index_2 =0
    for ind,ele in enumerate(l):
        if (ind != maxIndex):
            if (max_2 < ele):
                max_2 = ele
                index_2=ind
    return max_2,index_2




def get_pred_api(input_ids):
    headers = {
    "Content-Type": "application/json"
    }

    payload = {"signature_name": "serving_default","instances": [{"input_ids": input_ids}]}

    try:
        response = requests.post(TENSOR_SERVER_URL, json=payload, headers=headers)
        prediction = response.json()['predictions'][0]
    except Exception as error:
        raise Exception(error)

    return prediction

