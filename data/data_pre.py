'''
nuvi Lab 기존 csv 파일 train data로 쓰기위한 pre-processing 과정
'''
import pandas as pd
import datetime

prev_m = pd.read_csv('m_combined_data.csv',sep=',')
prev_e = pd.read_csv('e_combined_data.csv',sep=',')

#middle school
df = pd.DataFrame(prev_m)

new_df = df[['date','FoodId','food_supply','food_left']]

new_date = []
for i in range(len(new_df)):
	temp = "20"+str(int(new_df.loc[i]['date']))
	dt = datetime.datetime(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])).weekday()
	new_date.append(dt)

new_df['date'] = new_date
new_df.to_csv('./m_input.csv')

#elementary school
df = pd.DataFrame(prev_e)

new_df = df[['date','FoodId','food_supply','food_left']]

new_date = []
for i in range(len(new_df)):
	temp = "20"+str(int(new_df.loc[i]['date']))
	dt = datetime.datetime(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])).weekday()
	new_date.append(dt)

new_df['date'] = new_date
new_df.to_csv('./e_input.csv')