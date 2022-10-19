import pickle
from flask import Flask,request,jsonify,url_for,render_template,redirect
import numpy as np
import pandas as pd


app=Flask(__name__)
output=" "
##load the model
regmodel=pickle.load(open("regression.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
@app.route("/")
def home():
    return render_template('home.html')




@app.route('/result01')
def result01():
    return render_template('valve.html',resultk=output)

@app.route('/pridict',methods=['POST','GET'])
def pridict():
    

    data01=float(request.form['CRIM'])
    data02=float(request.form['ZN'])
    data03=float(request.form['INDUS'])
    data04=float(request.form['CHAS'])
    data05=float(request.form['NOX'])
    data06=float(request.form['RM'])
    data07=float(request.form['AGE'])
    data08=float(request.form['DIS'])
    data09=float(request.form['RAD'])
    data010=float(request.form['TAX'])
    data011=float(request.form['PTRATIO'])
    data012=float(request.form['B'])
    data013=float(request.form['LSTAT'])
    print(data011,data012,data013)
    list=[data01,data02,data03,data04,data05,data06,data07,data08,data09,data010,data011,data012,data013]
    deo=np.array(list).reshape(1,-1)
    new1_data=scaler.transform(deo)
    output=regmodel.predict(new1_data)[0]
    res=result01
    ##print(output)
    ##return jsonify(output[0])
    return render_template("home.html",prediction_text="the house pridiction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
