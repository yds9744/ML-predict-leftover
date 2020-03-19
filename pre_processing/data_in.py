import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

if __name__=="__main__":
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
	data = pd.read_csv('./final_data.csv',na_values = {'food_supply':[],'food_left':[]},sep=',')
	df = pd.DataFrame(data)

	#drop out null data
	xy = np.array(df.dropna(), dtype=np.int32)
	x_data = xy[:,1:3]
	y_data = xy[:,3:5]
	
	#transform [date, foodId] -> categorical data
	temp=[]
	for i in range(len(x_data)):
		temp.append([x_data[i][0]]+menu_id[x_data[i][1]])
	temp = np.array(temp)
	
	'''
	temp라는 array 형태가 [요일 재료 조리 메뉴]를 의미하는 값들로 채워져잇음
	요일 5개(주말 없으니까), 재료 5개,조리 2개, 메뉴6개니까 총 1*18짜리 벡터로 인코딩됨
	'''
	one_hot_e = OneHotEncoder()
	X = one_hot_e.fit_transform(temp).toarray()