from flask import Flask, render_template
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
pkl_name = 'LR_model.pkl'
model = joblib.load(pkl_name)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/info')
def info():
	t = [[0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,200]]
	t = np.array(t)
	
	ret = model.predict(t)
	
	return render_template('index.html',price = ret[0])
