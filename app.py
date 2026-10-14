from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path


__author__ = "rkowdeed"

app = Flask(__name__)
with (Path(__file__).resolve().parent / 'Kidney.pkl').open('rb') as model_file:
    model = pickle.load(model_file)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = pd.DataFrame(
            [[sg, htn, hemo, dm, al, appet, rc, pc]],
            columns=['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc'],
        )
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
