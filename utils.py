import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from models.LinearRegression import LinearRegression

np.random.seed(428)

def load_data():
	'''
	메뉴 id 읽어오는 부분
	딕셔너리에 dict[id] = ['재료','조리','메뉴'] 순으로 카테고리 value list 저장
	ex) menu_id[9] = [1	0 1] 이런 형태로 저장
	'''
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

	'''
	temp라는 array 형태가 [요일 재료 조리 메뉴]를 의미하는 값들로 채워져잇음
	요일 5개(주말 없으니까), 재료 5개,조리 2개, 메뉴6개니까 총 1*18짜리 벡터로 인코딩됨
	'''
	one_hot_e = OneHotEncoder().fit_transform(temp).toarray()
	x = np.hstack((one_hot_e, x[:,2].reshape(-1,1)))

	return x, y

def split_data(test_ratio):
	x, y = load_data()
	data = np.hstack((x, y.reshape(-1, 1)))

	np.random.shuffle(data)
	test_size = int(len(data)*test_ratio)
	test_x = data[:test_size, :-1]
	test_y = data[:test_size, -1]
	train_x = data[test_size:, :-1]
	train_y = data[test_size:, -1]

	return (test_x, test_y), (train_x, train_y)

def initialize(test_ratio):
    model = LinearRegression
    test_data, train_data = split_data(test_ratio)

    return test_data, train_data, model