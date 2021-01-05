# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask("__main__")


lr = joblib.load("model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

app.run(port=12345, debug=True)

@app.route('/predict/<int:age>/<string:sex>/<string:embarked>', methods=['GET'])
def predict(age,sex,embarked):
    if lr:
        try:
            query = pd.get_dummies(pd.DataFrame({
                "Age":age,
                "Sex":sex,
                "Embarked":embarked
            },index=[1]))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = lr.predict(query)

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

