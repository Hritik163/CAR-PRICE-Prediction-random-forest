import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
## import ridge regresor model and standard scaler pickle
random_forest_model=pickle.load(open('model/model.pkl','rb'))
standard_scaler=pickle.load(open('model/scaler.pkl','rb'))
#Route for home page
@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Car_Name = int(request.form.get('Car_Name'))
        Year = float(request.form.get('Year'))
        Present_Price = float(request.form.get('Present_Price'))
        Kms_Driven = float(request.form.get('Kms_Driven'))
        Fuel_Type = int(request.form.get('Fuel_Type'))
        Seller_Type = int(request.form.get('Seller_Type'))
        Transmission = int(request.form.get('Transmission'))
        Owner = int(request.form.get('Owner'))
        new_data_scaled = standard_scaler.transform([[Car_Name,Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]])
        result=random_forest_model.predict(new_data_scaled)

        return render_template('home.html', result=result)
    else:
        return render_template("home.html")

app.run(debug=True)