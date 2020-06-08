#!flaskapi/bin/python
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "" #(or "1" or "2")

from flask import Flask
from flask import request, jsonify
import predictor_bert as pred

app = Flask(__name__)

@app.route('/stressor',methods=['GET'])
def index():
    stressor = request.args.get('stressor', default=0, type=str) 
    res = pred.bert_predict(stressor)
    
    return res
    

if __name__ == '__main__':
    app.run(debug=True)
