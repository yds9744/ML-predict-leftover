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
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	req = request.form
	
	i=1
	cate_data = [[0,0,0,0],[1,1,1,1],[2,2,0,2],[3,3,0,3],[4,4,0,4],[0,0,0,5]]
	supp_data = []
	while "menu"+str(i) in req:
		cate_data.append([0]+encoded[req["menu"+str(i)]])
		supp_data.append([int(req["supply"+str(i)])])
		i+=1
	
	one_hot_e = OneHotEncoder().fit_transform(cate_data).toarray()
	
	x = np.hstack( (one_hot_e[6:,:], supp_data) )
	print(model.predict(x))

	ret = np.sum(model.predict(x))

	return render_template('index.html',price = round(ret,2))

@app.route('/auto', methods=['POST'])
def auto():
	global menu
	
	ret = json.dumps(menu)
	return ret
