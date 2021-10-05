import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Model
from tensorflow.keras import activations

base = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Regresao multiplas classes/files/games.csv')

nome_jogos = base.Name
# Drop Colunas
base = base.drop('Developer', axis=1)
base = base.drop('Name', axis=1)
base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)
base = base.drop('Other_Sales', axis=1)
# Drop Linhas vazias
base = base.dropna(axis=0)
base = base.loc[base.Global_Sales > 1.5]

previsores = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
global_Sales = base.iloc[:, 4].values

labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = ColumnTransformer(transformers=[(
    "OneHot", OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# Criadndo a rede neural
camada_entrada = Input(shape=(79,))
camada_oculta1 = Dense(
    units=50, activation=activations.sigmoid)(camada_entrada)
camada_oculta2 = Dense(
    units=50, activation=activations.sigmoid)(camada_oculta1)
camada_oculta3 = Dense(
    units=50, activation=activations.sigmoid)(camada_oculta2)
camada_saida = Dense(units=1, activation=activations.linear)(camada_oculta3)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida])

regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, [global_Sales], epochs=3000, batch_size=100)

previsao_global = regressor.predict(previsores)
