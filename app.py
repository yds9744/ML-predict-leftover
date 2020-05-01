from flask import Flask, render_template, request
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import numpy as np

app = Flask(__name__)
pkl_name = 'model.pkl'
model = joblib.load(pkl_name)

menu = []
encoded = dict()

scaler = MinMaxScaler()
scaler.fit([[7700, 21, 95], [0, 2, 34]])

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
		supp_data.append([int(supply[i]),15,60])
	
	one_hot_e = OneHotEncoder().fit_transform(cate_data).toarray()

	x = np.hstack( (one_hot_e[6:,:], scaler.transform(supp_data)) )
	#print(model.predict(x))

	ret = np.sum(model.predict(x))

	return str(round(ret,2))

@app.route('/auto', methods=['POST'])
def auto():
	global menu
	
	ret = json.dumps(menu)
	return ret

@app.route('/addmenu', methods = ['GET'])
def add():
	return render_template('add.html')

@app.route('/addmenu', methods = ['POST'])
def add_menu():
	global df
	global menu

	req = request.form
	
	for idx, row in df.iterrows():
		if row['음식명'] == req['name']:
			return "이미 존재하는 메뉴입니다."

	menu.append(req['name'])
	new_row = [df.loc[len(df)-1]['FoodId']+1, req['name'], req['cate1'], req['cate2'], req['cate3']] 
	df.loc[len(df)] = new_row
	encoded[req['name']] = [int(req['cate1']), int(req['cate2']), int(req['cate3'])]
	
	#writer = pd.ExcelWriter(, engine='xlsxwriter')
	df.to_excel('./data/menu.xlsx', index=False)

	return req['name']+"(이)가 추가되었습니다."

@app.route('/cate', methods = ['POST'])
def cate():
	cate_list = [['곡류','육류','해산물류','야채류','과일류'],['비튀김','튀김'],['밥','국','반찬','김치','후식','일품요리']]

	ret = json.dumps(cate_list)
	return ret
