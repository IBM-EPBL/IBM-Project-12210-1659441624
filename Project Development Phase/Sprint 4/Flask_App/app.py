from flask import Flask,render_template,request,redirect
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib
import scipy
import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "C1B-PMVbuqCxmK99GBmu84fR05FiT5pJreoURR-yzggu"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


app = Flask(__name__)
model = load_model('B2_IBM_model.tar.gb')

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == "POST":
        string = request.form['val']
        string = string.split(',')
        x_input = [eval(i) for i in string]
        

        sc = joblib.load("scaler.save") 

        x_input = sc.fit_transform(np.array(x_input).reshape(-1,1))

        x_input = np.array(x_input).reshape(1,-1)

        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,10,1))
        print(x_input.shape)

        model = load_model('crude_oil.h5')
        output = model.predict(x_input)
        print(output[0][0])

        val = sc.inverse_transform(output)
        
        payload_scoring = {"input_data": [{   "values": [[x_input]]    }]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/90e3b6e1-c3a0-448b-b61a-47392049e8f3/predictions?version=2022-11-12', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
        predictions = response_scoring.json()
        print(response_scoring.json())
        return render_template('index.html' , prediction = val[0][0])
    if request.method=="GET":
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)




