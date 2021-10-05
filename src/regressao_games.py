import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Model

base = pd.read_csv('/Users/es19237/Desktop/Deep Learning/Regresao multiplas classes/files/games.csv')

base  = base.drop('Other_Sales', axis=1)
base  = base.drop('Global_Sales', axis=1)
base  = base.drop('Developer', axis=1)
"""Quando o axis é 1 ele apaga a coluna e quando é 0 ele apaga por linha"""
base = base.dropna(axis = 0)

a= base.loc[base['NA_Sales']<1]

base = base.loc[base['NA_Sales']>1]
base = base.loc[base['EU_Sales']>1]

base['Name'].value_counts()

nome_jogos = base.Name

base  = base.drop('Name', axis=1)

#Fim do processamento

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:,4].values
venda_eu = base.iloc[:,5].values
venda_jp = base.iloc[:,6].values


#Transformando o texto em valores para a rede neural conseguir trabalhar
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])

#Transformando em dados de grid.
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

#Criadndo a rede neural mais robusta
camada_entrada = Input(shape=(61,))
#61+3/2 o 3 é qtde de saidas q querro
camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)

camada_saida1 = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation='linear')(camada_saida1)
camada_saida3 = Dense(units = 1, activation='linear')(camada_saida2)

regressor= Model(inputs=camada_entrada,outputs=[camada_saida1,camada_saida2,camada_saida3])

regressor.compile(optimizer='adam',loss='mse')
regressor.fit(previsores,[venda_na,venda_eu,venda_jp],epochs=10000,batch_size=100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)

previsao_na.mean()
venda_na.mean()

previsao_eu.mean()
venda_eu.mean()

previsao_jp.mean()
venda_jp.mean()





















