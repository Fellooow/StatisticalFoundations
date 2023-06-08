import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.model_selection import train_test_split

PATH = 'Housing.csv'
COLUMNS_FOR_FACTORISATION = {'price'}

df = pd.read_csv(PATH)
factorisation_table = {}

df.drop(
    [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea',
        'furnishingstatus'
    ],
    axis=1,
    inplace=True
)

for column in df.columns:
    if column in COLUMNS_FOR_FACTORISATION:
        df[column], table = pd.factorize(df[column])
        factorisation_table[column] = pd.DataFrame(
            columns=[column],
            data=table
        )

print(df)

sns.heatmap(
    round(
        abs(df.corr()),
        1
    ),
    annot=True
)
plt.show()

train_input, test_input, train_output, test_output = train_test_split(
    df.drop('price', axis=1),
    df['price'],
    test_size=0.2
)

model = tree.DecisionTreeClassifier()
model.fit(train_input, train_output)

predictions = model.predict(test_input)
confusion_matrix = sklearn.metrics.confusion_matrix(predictions, test_output)

sns.heatmap(
    confusion_matrix,
    annot=True
)
plt.title('Матррица сходства')
plt.show()

plt.figure(dpi=600)
tree.plot_tree(model)
plt.title('Дерево решений')
plt.show()