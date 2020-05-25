import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("xgbclassifier")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/cardio',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # arr = []
    # for i in request.form.values():
    #     arr.append(float(i))
    # print(arr)
            
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]

    output = " " if(model.predict(final_features) == 1) else "Not"
    return render_template('index.html', prediction_text='You are {} diagonised with Cardio disese'.format(output))

    return render_template("index.html", prediction_test = "You are dead!!")
if __name__ == "__main__":
    app.run(debug=True)