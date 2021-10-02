import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("C:/Users/user/Desktop/ML using Python/House.csv")
print(data.head())

fig = go.Figure(data=[go.Table(header=dict(values=list(data.columns),
fill_color='silver',align='left'),cells=dict
(values=[data.AvgAreaIncome,data.AvgAreaHouseAge,data.AvgAreaNumberofRooms,data.AvgAreaNumberofBedrooms,data.AreaPopulation,
         data.Price,data.Address]
,fill_color='wheat',align='center'))])
fig.update_layout(title="Housing Price Prediction Dataset")
fig.show()

data.info()
data.describe()

sns.pairplot(data)
sns.displot(data['Price'])
plt.show()
sns.heatmap(data.corr())
plt.show()

X = data[['AvgAreaIncome', 'AvgAreaHouseAge', 'AvgAreaNumberofRooms','AvgAreaNumberofBedrooms', 'AreaPopulation']]
Y = data['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train ,Y_train )

print(lm.intercept_)

coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff)

prediction = lm.predict(X_test)
print("Prediction  : ",prediction)

plt.scatter(Y_test,prediction)
sns.displot((Y_test-prediction),bins=50,color='green')
plt.show()

print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))

prediction = lm.predict(X_test)
print("Prediction  : ",prediction)

plt.scatter(Y_test,prediction)
sns.displot((Y_test-prediction),bins=50,color='green')
plt.show()

print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
