'''
nuvi Lab 기존 csv 파일 train data로 쓰기위한 pre-processing 과정
'''
import pandas as pd
import datetime

prev_m = pd.read_csv('m_combined_data.csv',sep=',')
prev_e = pd.read_csv('e_combined_data.csv',sep=',')
w_file = pd.read_csv('w_data.csv',sep=',',engine='python')

#weather dataframe and dictionary
w_df = pd.DataFrame(w_file)
w_df = w_df[['일시','기온','습도']]
w_dict = dict()

for i in range(len(w_df)):
	if w_df.loc[i]['일시'][-5:]=="12:00":
		k = w_df.loc[i]['일시'][2:4]+w_df.loc[i]['일시'][5:7]+w_df.loc[i]['일시'][8:10]
		w_dict[k] = [w_df.loc[i]['기온'],w_df.loc[i]['습도']]

#middle school
df = pd.DataFrame(prev_m)

new_df = df[['date','FoodId','food_supply','food_left']]

new_date = []
temperature = []
humidity = []
for i in range(len(new_df)):
	temp = "20"+str(int(new_df.loc[i]['date']))
	dt = datetime.datetime(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])).weekday()
	new_date.append(dt)
	temperature.append(w_dict[str(int(new_df.loc[i]['date']))][0])
	humidity.append(w_dict[str(int(new_df.loc[i]['date']))][1])

new_df['date'] = new_date
new_df['temperature'] = temperature
new_df['humidity'] = humidity
new_df.to_csv('./m_input.csv')

#elementary school
df = pd.DataFrame(prev_e)

new_df = df[['date','FoodId','food_supply','food_left']]

new_date = []
temperature = []
humidity = []
for i in range(len(new_df)):
	temp = "20"+str(int(new_df.loc[i]['date']))
	dt = datetime.datetime(int(temp[0:4]),int(temp[4:6]),int(temp[6:8])).weekday()
	new_date.append(dt)
	temperature.append(w_dict[str(int(new_df.loc[i]['date']))][0])
	humidity.append(w_dict[str(int(new_df.loc[i]['date']))][1])

new_df['date'] = new_date
new_df['temperature'] = temperature
new_df['humidity'] = humidity
new_df.to_csv('./e_input.csv')