from copyreg import pickle
from pyexpat import model
from unittest import result
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pred
app=Flask(__name__)
from pickle import load,dump
#model=load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template('prediction.html')
'''prediction=model.predict(X_test)
for i in range(len(prediction)):
    prediction[X]=1
#prediction function
def ValuePredictor(sym):
    #creating input data for the models
    input_data=np.array(X_test).reshape(1,-1)
    result=input_data.predict(X)
    return result[0]'''


@app.route("/result",methods=['POST'])
def result():
    if request.method=='POST':
        data1=request.form['symptom 1']
        data2=request.form['symptom 2']
        data3=request.form['symptom 3']
        data4=request.form['symptom 4']
        #arr=np.array([[data1,data2,data3,data4]])
        predw=pred.classify(data1,data2,data3,data4)
        #pred=model.predict(arr)
        #result=ValuePredictor(pred)
        return render_template('result.html',predwe=predw)
if(__name__=='__main__'):
    app.run(debug=True)