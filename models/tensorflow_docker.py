
import requests

TENSOR_SERVER_URL = "http://127.0.0.1:8501/v1/models/bertstressor:predict"
TENSOR_COVID_SERVER_URL = "http://127.0.0.1:8502/v1/models/bertcovidstressor:predict"

def get_pred_api(input_ids,url):
    headers = {
    "Content-Type": "application/json"
    }

    payload = {"signature_name": "serving_default","instances": [{"input_ids": input_ids}]}

    try:
        response = requests.post(url, json=payload, headers=headers)
        prediction = response.json()['predictions'][0]
    except Exception as error:
        raise Exception(error)

    return prediction