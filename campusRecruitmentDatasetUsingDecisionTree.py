import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
file_loc = '/home/tuesday/Dataset/Placement_Data_Full_Class.csv'
df = pd.read_csv(file_loc)

df['status'].replace("Placed",1,inplace=True)
df['status'].replace("Not Placed",0,inplace=True)
df['salary'].fillna(0,inplace=True)

features = ['ssc_p','hsc_p','degree_p','etest_p','status','mba_p']
X = df[features]
y = df['salary']
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)
salary_model = DecisionTreeRegressor(random_state = 1)

salary_model.fit(train_X,train_y)

val_pred = salary_model.predict(val_X)
mae = mean_absolute_error(val_pred,val_y)
print("Mean Absolute Error is "+mae)

