# Popbots-Flask-API-Stressor

The purpose of this API endpoint is to route queries to a controller which converts a string sentence into a correctly tokenized BERT 32 tokens array and then query a BERT model served with Tensorflow Serving Docker container. 

## 1. Deploy 


1.1. Create a python3.7 venv (see online tutorials) with the name flaskapi ( you can configure a different name but you'll need to change the venv name in app.py at line 1 and configure the gitignore)
- Activate the venv by doing: 
    > source "venv_name"/bin/activate
- install all the required packages by navigating into the popbots folder and running 
    > pip3 install -r pip_requirements.txt

- run via the bash command

    > python app.py

Note: if you are using pm2 to manage the services you will need to specify the python interpreter

> pm2 start app.py --interpreter ./flaskapi/bin/python3 --name flask_classifiers_api


## 2. Routes

The is only 2 routes in this API: 

/classifier/stressor which given the param string 'stressor' will return the category0 to 8 sorted by probability from higher to lower 

/classifier/covid which given the param string 'stressor' will return if it is covid related or not. 

A typical query looks like http://{CLASSIFIER_IP_ADDRESS}/classifier/stressor?stressor="this upcoming deadline"



## 5. The docker container for Tensorflow serving


The command to execute the docker container is as follow: 

Here are the instructions to install the tensorflow docker serving container
https://www.tensorflow.org/tfx/serving/docker


```bash
sudo docker run -p 8501:8501 --name tfserving_bertstressor --mount type=bind,source=//home/ec2-user/models/,target=/models/bertstressor -e NVIDIA_VISIBLE_DEVICES=none -e MODEL_NAME=bertstressor -t tensorflow/serving:latest-gpu --enable_batching=false &
```
Make sure that this runs with : 
```
curl -X POST   http://localhost:8501/v1/models/bertstressor:predict   -H 'Content-Type: application/json'   -H 'Postman-Token: bd0bb5d4-6409-4de5-9bc6-8ee45d69108a'   -H 'cache-control: no-cache'   -d '{"signature_name": "serving_default", "instances": [{"input_ids": [101, 2147, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]}'
```


To execute the covid model : 
```bash
sudo docker run -p 8502:8501 --name tfserving_bertcovidstressor --mount type=bind,source=//home/ec2-user/covid_model/,target=/models/bertcovidstressor -e NVIDIA_VISIBLE_DEVICES=none -e MODEL_NAME=bertcovidstressor -t tensorflow/serving:latest-gpu --enable_batching=false &

# might be 8502:8502 not sure
```

You can make sure that this runs with the following query 

```
curl -X POST   http://localhost:8502/v1/models/bertcovidstressor:predict   -H 'Content-Type: application/json'   -H 'Postman-Token: bd0bb5d4-6409-4de5-9bc6-8ee45d69108a'   -H 'cache-control: no-cache'   -d '{"signature_name": "serving_default", "instances": [{"input_ids": [101, 2147, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]}'

```


The both of these models files host a pb format Tensorflow estimator: 

> /home/ec2-user/models/ # for the stressor classifier 

> /home/ec2-user/covid_model/ # for the covid classifier

these are obtained via the Popbots-mturk-HITS/bert-pipeline jupyter notebook by the function export_model