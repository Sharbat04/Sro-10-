import pandas as pd

data = pd.read_csv('Test.csv')

print(data.head())

X = data[['Temperature', 'Wind speed']]
y = data['Gusts of wind']

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

base_model = DecisionTreeRegressor(max_depth=3)

adaboost_model = AdaBoostRegressor(base_model, n_estimators=50, random_state=42)

adaboost_model.fit(X, y)

X_new = [[22.0, 1.5]]  

predictions = adaboost_model.predict(X_new)
print("Predicted Gusts of wind:", predictions)
