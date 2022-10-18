import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
##load the model
regmodel=pickle.load(open("regression.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/pridict_api',methods=['POST'])
def pridict_api():
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,13))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,13))
    output=regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
