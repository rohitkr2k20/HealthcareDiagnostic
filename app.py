from flask import Flask, render_template, redirect, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import joblib
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf


model = joblib.load('gradientBoostingClass.sav')
modelCovid = load_model('model_adv.h5')

std_scaler = StandardScaler()
data = pd.read_csv("hccdatacompletebalanced.csv")
data = data.values
data = pd.DataFrame(data)
data = data.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47], axis=1) 
data = data.values
dataX = data[:, :-1]
Y = data[:, -1]
std_scaler.fit(dataX)

def predicthcc(X):
    X = std_scaler.transform(X)
    ans = model.predict(X)
    return ans[0]

def predictCovid(path):
	
	img = tf.keras.utils.load_img(path, target_size=(224,224))
	img = tf.keras.utils.img_to_array(img) / 255.0
	img = np.expand_dims(img, axis=0)
	ans = (modelCovid.predict(img) > 0.5).astype("int32")
	return ans[0][0]

app = Flask(__name__)

@app.route('/home')
@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/hcc1')
def hcc1():
	return render_template("hcc1.html")

@app.route('/covid')
def covid():
	return render_template("covid.html")

@app.route('/hcc1', methods=['POST'])
def hcc1Solve():
	ans = 0
	if request.method == 'POST':
		f1 = float(request.form['f1'])
		f2 = float(request.form['f2'])
		f3 = float(request.form['f3'])
		f4 = float(request.form['f4'])
		f5 = float(request.form['f5'])

		ans = predicthcc([[f5, f4, f2, f1, f3]])

	ans_str = ""
	if ans:
		ans_str = "Hurray!!! You have High Survival Chances from Liver Cancer"
	else:
		ans_str = "You have low Survival Chances from Liver Cancer"

	return render_template('hcc1.html', your_res_hcc1=ans_str)


@app.route('/covid', methods=['POST'])
def covidSolve():
	ans = 0
	if request.method == 'POST':
		f = request.files['userfile']
		path = './static/{}'.format(f.filename)
		f.save(path)

		ans = predictCovid(path)
	
	ans_str = ""
	if ans:
		ans_str = "Hurray!!! You are Covid Negative"
	else:
		ans_str = "You are Covid Positive!!! Please Quarantine Yourself"

	return render_template('covid.html', your_res_covid=ans_str)


if __name__ == '__main__':

	app.run(debug=True)
