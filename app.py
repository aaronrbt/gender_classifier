import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
from tensorflow.keras.models import load_model
from genderclassifier.util import *
#import logging
#ogging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
model = load_model('model/final_lstm.h5', custom_objects={"f1_metric": f1_metric})

MAXLEN = 25

def unit_prediction(ipt:str):
    try:
        ipt_array = process_unit_ipt(ipt,MAXLEN)
        assert not isinstance(ipt_array, int), "AssertionError"
        pred = model.predict(ipt_array,batch_size=32)
        pred_code = (pred>0.5).astype('int32').flatten()[0]
        prediction = code_to_label.get(pred_code)
        return prediction
    except Exception as e:
        print(f"Wrong Input : {ipt} | {e}")
        return "NA (Invalid Name Detected)"


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/result',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    ipt = [x for x in request.form.values()][0]
    prediction = unit_prediction(ipt)
    return render_template('index.html', prediction_text=f"Predicted Gender: {prediction}")

@app.route("/api", methods=['POST'])
def api():
    ipt = request.get_json().get("First Name",'')
    msg = "NA"
    if request.headers['Content-Type'] == 'application/json':
        prediction = unit_prediction(ipt)
        if ipt == '':
            status_code = 406
            msg = 'No Name Detected.'
        elif prediction not in ['M','F']:
            status_code = 400
            msg = prediction
        else:
            status_code = 200
            msg = prediction
    opt = {"Predicted Gender": f"{msg}"}
    return make_response(jsonify(opt), status_code)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)