# =============================================================================
# Project Name: BayesGuard
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv("emails.csv") #read_csv works but read.csv is not a command beacuse read alone is not syntax
#data.head()
#data.info()
data.drop(columns=['Email No.'], inplace = True)


data.isna().sum() #should total up the number of empty rows

#data.shape

x = data.iloc[: ,0:3000]  
y = data.iloc[: , 3000]

#(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)  #80%training, 20% test 

model = GaussianNB()    #first nb on scikitlearn website
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
#(np.array(y_test),y_prediction)

print("accuracy : " , np.round(accuracy_score(y_test, y_prediction),4)*100)

#cross_validation = cross_val_score(model,x_train,y_train,cv = 10)
#print(cross_validation.mean() * 100)