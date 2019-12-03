#!flaskapi/bin/python
from flask import Flask
from flask import request, jsonify
import predictor as pred

app = Flask(__name__)

@app.route('/stressor',methods=['GET'])
def index():
    stressor = request.args.get('stressor', default=0, type=str) 
    res = pred.lstm_predict(stressor)
    
    return res
    

if __name__ == '__main__':
    app.run(debug=True)