import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sys
import logging

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

app = Flask(__name__, template_folder='templates')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
model = pickle.load(open('model.pkl', 'rb'))

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
	For rendering results on HTML GUI
	'''
    print('Enter input values')
    int_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([int_features])
    final_features.columns=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11',
	'V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    if output==1:
        return render_template('index.html', prediction_text='Transaction seems to be fraudulent')
    elif output==0:
        return render_template('index.html', prediction_text='Transaction seems to be non-fraudulent')
    else:
        return render_template('index.html', prediction_text='Please enter input values') 
     


if __name__ == "__main__":
    app.run(debug=True)
    
