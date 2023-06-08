import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

PATH = 'cars.csv'
dataset = pd.read_csv(PATH)

for column in dataset:
    if type(dataset[column][0]) is str:
        dataset[column] = pd.factorize(dataset[column])[0]

print(dataset)
sns.heatmap(
    round(
        abs(dataset.corr()),
        1,
    ),
    annot=True,
)

plt.plot(dataset)
plt.show()

train_input, test_input, train_output, test_output = train_test_split(
    dataset.drop('fuel', axis=1),
    dataset['fuel'],
    test_size=0.3
)

model = GaussianNB()
model.fit(train_input, train_output)

predictions = model.predict(test_input)
print(predictions[:10])
accuracy = metrics.accuracy_score(predictions, test_output)

print(test_input)
print(f'Точность модели на тестовом участке = {accuracy}')
