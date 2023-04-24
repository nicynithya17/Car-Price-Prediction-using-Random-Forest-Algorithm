import pandas as pd


df=pd.read_csv('car_price_data.csv')

df.shape

print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
df.isnull().sum()
df.describe()

f_data=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

f_data.head()

f_data['Current Year']=2020

f_data['no_year']=f_data['Current Year']- f_data['Year']

f_data.head()

f_data.drop(['Year'],axis=1,inplace=True)

f_data.head()

f_data=pd.get_dummies(f_data,drop_frst=True)

f_data=f_data.drop(['Current Year'],axis=1)

f_data.head()

f_data.corr()

import seaborn as sns

sns.pairplot(f_data)

import matplotlib.pyplot as plt
import seaborn as sns


cmatrix = df.corr()
top_corr_features = cmatrix.index
plt.figure(figsize=(20,20))

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X=f_data.iloc[:,1:]
y=f_data.iloc[:,0]


X['Owner'].unique()

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)


fm = pd.Series(model.feature_importances_, index=X.columns)
fm.nlargest(5).plot(kind='barh')
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


import numpy as np
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()

estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(estimators)

from sklearn.model_selection import RandomizedSearchCV


estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

maxf = ['auto', 'sqrt']

maxd = [int(x) for x in np.linspace(5, 30, num = 6)]


minsplit = [2, 5, 10, 15, 100]


minsamleaf = [1, 2, 5, 10]

rgrid = {'estimators': estimators,
              'maxf': maxf,
              'maxd': maxd,
              'minsplit': minsplit,
              'minsamleaf': minsamleaf}

print(rgrid)

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rgrid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

rf_random.best_params_

rf_random.best_score_

predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)

from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


import pickle

file = open('random_forest_regression_model.pkl', 'wb')

pickle.dump(rf_random, file)
