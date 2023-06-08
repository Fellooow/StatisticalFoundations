import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 40)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('darkgrid')
import plotly.express as px
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

'''Loading data and cleaning data'''
PATH = 'cars_ds_final.csv'
dataset = pd.read_csv(PATH)
dataset.info()
# Важные особенности набора данных
l_D = len(dataset)
c_m = len(dataset.Make.unique())
c_c = len(dataset.Model.unique())
n_f = len(dataset.columns)
fig = px.bar(x=['Observations', "Makers", 'Models', 'Features'], y=[l_D, c_m, c_c, n_f], width=800, height=400)
fig.update_layout(
    title="Dataset Statistics",
    xaxis_title="",
    yaxis_title="Counts",
    font=dict(
        size=16,
    )
)

fig.show()


# Выбор подмножества полезных функций, очистка данных для извлечения полезной информации
dataset['car'] = dataset.Make + ' ' + dataset.Model
c = ['Make', 'Model', 'car', 'Body_Type', 'Fuel_Type', 'Fuel_System', 'Type', 'Drivetrain',
     'Ex-Showroom_Price', 'Displacement', 'Cylinders',
     'ARAI_Certified_Mileage', 'Power', 'Torque', 'Fuel_Tank_Capacity', 'Height', 'Doors',
     'Seating_Capacity']
dataset_full = dataset.copy()
dataset['Ex-Showroom_Price'] = dataset['Ex-Showroom_Price'].str.replace('Rs. ', '', regex=False)
dataset['Ex-Showroom_Price'] = dataset['Ex-Showroom_Price'].str.replace(',', '', regex=False)
dataset['Ex-Showroom_Price'] = dataset['Ex-Showroom_Price'].astype(int)
dataset = dataset[c]
dataset = dataset[~dataset.ARAI_Certified_Mileage.isnull()]
dataset = dataset[~dataset.Make.isnull()]
# dataset = dataset[~dataset.Width.isnull()]
dataset = dataset[~dataset.Cylinders.isnull()]
# dataset = dataset[~dataset.Wheelbase.isnull()]
dataset = dataset[~dataset['Fuel_Tank_Capacity'].isnull()]
dataset = dataset[~dataset['Seating_Capacity'].isnull()]
dataset = dataset[~dataset['Torque'].isnull()]
dataset['Height'] = dataset['Height'].str.replace(' mm', '', regex=False).astype(float)
# dataset['Length'] = dataset['Length'].str.replace(' mm', '', regex=False).astype(float)
# dataset['Width'] = dataset['Width'].str.replace(' mm', '', regex=False).astype(float)
# dataset['Wheelbase'] = dataset['Wheelbase'].str.replace(' mm', '', regex=False).astype(float)
dataset['Fuel_Tank_Capacity'] = dataset['Fuel_Tank_Capacity'].str.replace(' litres', '', regex=False).astype(float)
dataset['Displacement'] = dataset['Displacement'].str.replace(' cc', '', regex=False)
dataset.loc[dataset.ARAI_Certified_Mileage == '9.8-10.0 km/litre', 'ARAI_Certified_Mileage'] = '10'
dataset.loc[dataset.ARAI_Certified_Mileage == '10kmpl km/litre', 'ARAI_Certified_Mileage'] = '10'
dataset['ARAI_Certified_Mileage'] = dataset['ARAI_Certified_Mileage'].str.replace(' km/litre', '', regex=False).astype(
    float)
# dataset.Number_of_Airbags.fillna(0, inplace=True)
dataset['price'] = dataset['Ex-Showroom_Price'] * 0.014
dataset.drop(columns='Ex-Showroom_Price', inplace=True)
dataset.price = dataset.price.astype(int)
HP = dataset.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
HP = HP.apply(lambda x: round(x, 2))
TQ = dataset.Torque.str.extract(r'(\d{1,4}).*').astype(int)
TQ = TQ.apply(lambda x: round(x, 2))
dataset.Torque = TQ
dataset.Power = HP
dataset.Doors = dataset.Doors.astype(int)
dataset.Seating_Capacity = dataset.Seating_Capacity.astype(int)
# dataset.Number_of_Airbags = dataset.Number_of_Airbags.astype(int)
dataset.Displacement = dataset.Displacement.astype(int)
dataset.Cylinders = dataset.Cylinders.astype(int)
dataset.columns = ['make', 'model', 'car', 'body_type', 'fuel_type', 'fuel_system', 'type', 'drivetrain',
                   'displacement', 'cylinders',
                   'mileage', 'power', 'torque', 'fuel_tank', 'height', 'doors', 'seats', 'price']

'''Exploratory Data analysis'''
# print(dataset[dataset.model =='Corolla Altis'])

# Check the price distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
sns.histplot(data=dataset, x='price', bins=50, alpha=.6, color='darkblue', ax=ax1)
ax12 = ax1.twinx()
sns.kdeplot(data=dataset, x='price', alpha=.2, fill=True, color="#254b7f", ax=ax12, linewidth=0)
ax12.grid()
ax1.set_title('Гистограмма данных о ценах на автомобили', fontsize=16)
ax1.set_xlabel('')
logbins = np.logspace(np.log10(3000), np.log10(744944.578), 50)
sns.histplot(data=dataset, x='price', bins=logbins, alpha=.6, color='darkblue', ax=ax2)
ax2.set_title('Гистограмма данных о ценах на автомобили (логарифмическая шкала)', fontsize=16)
ax2.set_xscale('log')
ax22 = ax2.twinx()
ax22.grid()
sns.kdeplot(data=dataset, x='price', alpha=.2, fill=True, color="#254b7f", ax=ax22, log_scale=True, linewidth=0)
ax2.set_xlabel('Price (log)', fontsize=14)
ax22.set_xticks((800, 1000, 10000, 100000, 1000000))
ax2.xaxis.set_tick_params(labelsize=14)
ax1.xaxis.set_tick_params(labelsize=14)
# plt.show()

# Box plot of prices
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataset, x='price', width=.3, color='blue', hue='fuel_type')
plt.title('Блочная диаграмма цены', fontsize=18)
plt.xticks([i for i in range(0, 800000, 100000)], [f'{i:,}$' for i in range(0, 800000, 100000)], fontsize=14)
plt.xlabel('price', fontsize=14)
# plt.show()

# Body types
plt.figure(figsize=(16, 7))
sns.countplot(data=dataset, y='body_type', alpha=.6, color='darkblue')
plt.title('Автомобили по типу кузова', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('')
plt.ylabel('')
# plt.show()

# Price of every body type
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataset, x='price', y='body_type', palette='viridis')
plt.title('Блочная диаграмма цены каждого типа кузова', fontsize=18)
plt.ylabel('')
plt.yticks(fontsize=14)
plt.xticks([i for i in range(0, 800000, 100000)], [f'{i:,}$' for i in range(0, 800000, 100000)], fontsize=14)
# plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(data=dataset, x='power', y='price', hue='body_type', palette='viridis', alpha=.89, s=120)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('power', fontsize=14)
plt.ylabel('price', fontsize=14)
plt.title('Соотношение между мощностью и ценой', fontsize=20)
# plt.show()

# Pearson correlation grid
plt.figure(figsize=(22,8))
sns.heatmap(dataset.corr(), annot=True, fmt='.2%')
plt.title('Корреляция между различными величинами',fontsize=20)
plt.xticks(fontsize=14, rotation=320)
plt.yticks(fontsize=14)
plt.show()


# Подгонка набора функций для создания 8 кластеров автомобилей
dataset = dataset[dataset.price < 60000]
num_cols = [i for i in dataset.columns if dataset[i].dtype != 'object']

# Подгонка модели кластеризации K-средних с 10 кластерами и добавление столбца кластера в набор данных
km = KMeans(n_clusters=8, n_init=20, max_iter=400, random_state=0)
clusters = km.fit_predict(dataset[num_cols])
dataset['cluster'] = clusters
dataset.cluster = (dataset.cluster + 1).astype('object')
print(dataset.sample(5))

# Cколько автомобилей существует в каждом кластере
plt.figure(figsize=(14, 6))
sns.countplot(data=dataset, x='cluster', palette='viridis', order=dataset.cluster.value_counts().index)
# plt.yticks([i for i in range(0,65000,5000)])
plt.title('Number of cars in each cluster', fontsize=18)
plt.xlabel('Cluster', fontsize=16)
plt.ylabel('Number of cars', fontsize=16)
plt.xticks(fontsize=14)
# plt.show()


# Price vs power with clustering
plt.figure(figsize=(10,8))
sns.scatterplot(data=dataset, y='price', x='power',s=120,hue='cluster',palette='viridis')
plt.legend(ncol=4)
plt.title('Scatter plot of price and horsepower with clusters predicted', fontsize=18)
plt.xlabel('power',fontsize=16)
plt.ylabel('price',fontsize=16)
plt.show()

# Average prices of each cluster
plt.figure(figsize=(14,6))
sns.barplot(data=dataset, x= 'cluster', ci= 'sd', y= 'price', palette='viridis',order=dataset.groupby('cluster')['price'].mean().sort_values(ascending=False).index)
plt.yticks([i for i in range(0,65000,5000)])
plt.title('Average price of each cluster',fontsize=20)
plt.xlabel('Cluster',fontsize=16)
plt.ylabel('Avg car price', fontsize=16)
plt.xticks(fontsize=14)
plt.show()

#
df_c = dataset[dataset.cluster.isin([1,5])]
plt.figure(figsize=(8,6))
sns.countplot(data=df_c,x='body_type',palette='viridis')
plt.xlabel('Body type',fontsize=16)
plt.ylabel('Count of variants',fontsize=16)
plt.title('count of each body type in the targeted clusters (including variants)',fontsize=14)
plt.show()


inertia = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init= 'k-means++')
    k_means.fit_predict(dataset[num_cols], dataset.drop('model', axis=1))
    inertia.append(k_means.inertia_)

sns.set_style('darkgrid')
sns.scatterplot(
    x=[x for x in range(1, 11)],
    y=inertia
    )
plt.title('График зависимости')
plt.xlabel('Количество кластеров')
plt.ylabel('Внутри-кластерная сумма расстояний')
plt.show()
# BODIES = ["Sedan", "SUV"]
# COLUMNS_FOR_FACTORISATION = ["Make", "Model", "Ex-Showroom_Price"]
# NUMERIC_COLUMNS = ["Power", "Seating_Capacity", "Average_Fuel_Consumption"]
#
# factorization_table = {}
# dataset = dataset.loc[
#     dataset['Body_Type'].isin(
#         BODIES
#     )
# ]
# dataset['Car'] = dataset['Make'] + ' ' + dataset['Model']
# dataset.drop('Make', axis=1)
# dataset.drop('Model', axis=1)
#
#
#
# dataset = dataset[
#     [
#         "Car",
#         "Power",
#         "Torque",
#         "Seating_Capacity",
#         "Average_Fuel_Consumption",
#     ]
# ]
#
# for column in dataset.columns:
#     if column in COLUMNS_FOR_FACTORISATION:
#         dataset[column], table = pd.factorize(dataset[column])
#         factorization_table[column] = pd.DataFrame(
#             columns=[column],
#             data=table
#         )
#
#     if column in NUMERIC_COLUMNS:
#         dataset[column] = pd.to_numeric(dataset[column])
#         dataset.index = [index for index in range(len(dataset))]
#         factorization_table = {}
#         dataset = dataset.loc[
#             dataset['Bodies'].isin(
#                 BODIES
#             )
#         ]
# print(dataset)
#
# inertia = []
# for i in range(1, 11):
#     k_means = KMeans(n_clusters=i, init='k-means++')
#     k_means.fit(
#         dataset.drop("Car",axis=1)
#     )
#     inertia.append(k_means.inertia_)
#
# sns.set_style('darkgrid')
# sns.scatterplot(
#     x=[x for x in range(1, 11)],
#     y=inertia,
# )
# plt.title('График зависимости')
# plt.xlabel('Количество кластеров')
# plt.ylabel('Внутри-кластерная сумма расстояний')
#
# CLUSTERS = 4
# model = KMeans(
#     n_clusters=CLUSTERS
# )
# model.fit(
#     dataset.drop(
#         "Car",
#         axis=1,
#     )
# )
# clusters = pd.DataFrame(
#     columns=dataset.columns.drop('Car'),
#     data=model.cluster_centers_
# )
# clusters["Amount"] = np.unique(
#     model.labels_,
#     return_counts=True
# )[1]
#
# print(clusters)
