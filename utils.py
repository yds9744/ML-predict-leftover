import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from models.LinearRegression import LinearRegression
from models.KnnRegression import KnnRegression
from models.DecisionTreeRegression import DecisionTreeRegression
from models.RandomForestRegression import RandomForestRegression

np.random.seed(428)
data = []

def load_data():
	global data
	#menu id dictionary data (structure: {menu_id[idx]: [1,2,3]})
	menu_id = dict()
	menu_buff = pd.read_excel('./data/menu.xlsx')
	df = pd.DataFrame(menu_buff)

	for idx, row in df.iterrows():
		menu_id[int(row['FoodId'])] = [row['재료'],row['조리'],row['메뉴']]

	#food supply/left data in np.array
	data = pd.read_csv('./data/final_data.csv',na_values = {'food_supply':[],'food_left':[]},sep=',')
	df = pd.DataFrame(data)

	#drop out null data
	xy = np.array(df.dropna(), dtype=np.int32)
	x = xy[:,1:4]
	y = xy[:,4]

	#transform [date, foodId] -> categorical data
	temp=[]
	for i in range(len(x)):
		temp.append([x[i][0]]+menu_id[x[i][1]])
	temp = np.array(temp)

	numeric = np.hstack((x[:,2].reshape(-1,1),xy[:,-2:]))
	
	#One-Hot encoding categorical data
	one_hot_e = OneHotEncoder().fit_transform(temp).toarray()
	x = np.hstack((one_hot_e, numeric))
	#x = np.hstack((one_hot_e, x[:,2].reshape(-1,1)))
	data = np.hstack((x, y.reshape(-1, 1)))

# split train set / test set
def split_data(test_ratio):
	if len(data)==0: load_data()

	np.random.shuffle(data)
	test_size = int(len(data)*test_ratio)
	test_x = data[:test_size, :-1]
	test_y = data[:test_size, -1]
	train_x = data[test_size:, :-1]
	train_y = data[test_size:, -1]

	return test_x, test_y, train_x, train_y

def initialize(test_ratio, model_name):
	test_x, test_y, train_x, train_y = split_data(test_ratio)
	model = None

	if model_name == 'LinearRegression':
		model = LinearRegression()
	elif model_name == 'KnnRegression':
		test_x = np.hstack((test_x[:, :-3] * 10000, test_x[:, -3:]))
		train_x = np.hstack((train_x[:, :-3] * 10000, train_x[:, -3:]))
		model = KnnRegression()
	elif model_name == 'DecisionTreeRegression':
		model = DecisionTreeRegression()
	elif model_name == 'RandomForestRegression':
		model = RandomForestRegression()
	else:
		raise NotImplementedError

	return (test_x, test_y), (train_x, train_y), model