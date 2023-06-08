import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.regression import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



PATH = 'Housing.csv'
dataset = pd.read_csv(PATH)

dataset = dataset.drop('mainroad', axis=1)
dataset = dataset.drop('guestroom', axis=1)
dataset = dataset.drop('basement', axis=1)
dataset = dataset.drop('hotwaterheating', axis=1)
dataset = dataset.drop('airconditioning', axis=1)
dataset = dataset.drop('prefarea', axis=1)
dataset = dataset.drop('furnishingstatus', axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    dataset.drop('area', axis=1),
    dataset['area'],
    test_size=0.2,
    random_state=0
)

x_train = sm.add_constant(x_train)
sm_ols = linear_model.OLS(y_train, x_train)
sm_model = sm_ols.fit()

print(sm_model.summary())

x_test = sm.add_constant(x_test)
y_pred = sm_model.predict(x_test)

df = pd.DataFrame({'Актуальные значения': y_test,
                   'Предсказанные значения': y_pred})
print(df)


print('Средняя абсолютная ошибка (MAE): ',
      metrics.mean_absolute_error(y_test, y_pred))
print('Cредняя квадратичная ошибка (MSE): ',
      metrics.mean_squared_error(y_test, y_pred))
print('Квадратный корень средней квадратичной ошибки (RMSE): ',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(20, 10))
matrix = np.triu(dataset.corr())
sns.heatmap(dataset.corr().abs(), annot=True, mask=matrix)
plt.show()

#######
sns.lineplot(
    x=dataset['area'],
    y=dataset['price']
)

plt.title('График зависимости')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

x = np.array(dataset['area']).reshape(-1, 1)
y = np.array(dataset['price']).reshape(-1, 1)

DEGREES = 2
regression = make_pipeline(PolynomialFeatures(DEGREES), LinearRegression())
regression.fit(x, y)
predictions = regression.predict(x)
mean_squared_error = np.mean((predictions - np.array(y)) ** 2)
print(f'Среднеквадратичная ошибка: {mean_squared_error}')

sns.lineplot(
    x=dataset['area'],
    y=dataset['price'],
    linestyle='solid'
)

sns.lineplot(
    x=dataset['area'],
    y=predictions.reshape(-1),
    linestyle='dotted'
)
plt.title(
    'График зависимости\n\n'
    'Сплошная линия - эталонные значения\n'
    'Точечная линия - предсказания регрессии'
)
plt.xlabel('Area')
plt.show()
plt.ylabel('Price')

x_parameters = np.append(
    regression['linearregression'].intercept_[0],
    regression['linearregression'].coef_[0][1:]
)
print(x_parameters)
