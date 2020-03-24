from flask import Flask, render_template, request
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import json
import numpy as np



app = Flask(__name__)
pkl_name = 'LR_model.pkl'
model = joblib.load(pkl_name)

menu = []
encoded = dict()

menu_buff = pd.read_excel('./data/menu.xlsx')
df = pd.DataFrame(menu_buff)
for idx, row in df.iterrows():
	menu.append(row['음식명'])
	encoded[row['음식명']] = [row['재료'],row['조리'],row['메뉴']]

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
	return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict_result():
	req = request.form

	name = eval(req['name'])
	supply = eval(req['supply'])

	cate_data = [[0,0,0,0],[1,1,1,1],[2,2,0,2],[3,3,0,3],[4,4,0,4],[0,0,0,5]]
	supp_data = []
	for i in range(len(name)):
		cate_data.append([0]+encoded[name[i]])
		supp_data.append([int(supply[i])])
	
	one_hot_e = OneHotEncoder().fit_transform(cate_data).toarray()

	x = np.hstack( (one_hot_e[6:,:], supp_data) )
	#print(model.predict(x))

	ret = np.sum(model.predict(x))

	return str(round(ret,2))

@app.route('/auto', methods=['POST'])
def auto():
	global menu
	
	ret = json.dumps(menu)
	return ret
