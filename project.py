import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.impute import KNNImputer

data = pd.read_csv('C:\\Users\\natal\\Desktop\\project_data.csv',header=0,sep='\t')
enc = LabelEncoder()
data['zgrzytanie'] = enc.fit_transform(data['zgrzytanie'])
data['zaciskanie'] = enc.fit_transform(data['zaciskanie'])
data['sztywnosc'] = enc.fit_transform(data['sztywnosc'])
data['ograniczone otwieranie'] = enc.fit_transform(data['ograniczone otwieranie'])
data['bol miesni'] = enc.fit_transform(data['bol miesni'])
data['przygryzanie'] = enc.fit_transform(data['przygryzanie'])
data['cwiczenia'] = enc.fit_transform(data['cwiczenia'])
data['szyna'] = enc.fit_transform(data['szyna'])
data['starcie-przednie'] = enc.fit_transform(data['starcie-przednie'])
data['starcie-boczne'] = enc.fit_transform(data['starcie-boczne'])
data['ubytki klinowe'] = enc.fit_transform(data['ubytki klinowe'])
data['pekniecia szkliwa'] = enc.fit_transform(data['pekniecia szkliwa'])
data['impresje jezyka'] = enc.fit_transform(data['impresje jezyka'])
data['linea alba'] = enc.fit_transform(data['linea alba'])
data['przerost zwaczy'] = enc.fit_transform(data['przerost zwaczy'])
data['tkliwosc miesni'] = enc.fit_transform(data['tkliwosc miesni'])
data['plec'] = enc.fit_transform(data['plec'])
data['API'] = data['API'].str[:-1]
data['SBI'] = data['SBI'].str[:-1]
#print(data.head())
column_names = ['16-B','16-P','11-B','11-P','24-B','24-P','36-B','36-P','31-B','31-P','44-B','44-P']
y = data[['44-P']]
X = data.drop(columns=column_names)
#print(X)
#msno.matrix(X)
#plt.show()
#brak danych w ostatnich 12 kolumnach
X1 = X.iloc[:,1:57]
#print(X1)

X1 = X1.iloc[:,2:40]
column_names_forP = ['PPD - 16 B','PPD - 16 P','PPD - 11 B','PPD - 11 P','PPD - 24 B','PPD - 24 P','PPD - 36 B','PPD - 36 P','PPD - 31 B','PPD - 31 P','PPD - 44','API','SBI']
#X1 = data[['API','SBI','PI - 44','GI - 44','PPD - 44','TWI - 44 suma','zgrzytanie','zaciskanie','sztywnosc','ograniczone otwieranie','bol miesni',
#           'przygryzanie','cwiczenia','szyna','starcie-przednie','starcie-boczne','ubytki klinowe','impresje jezyka','linea alba','przerost zwaczy','tkliwosc miesni']]
X1 = X1.drop(columns=column_names_forP)
print(X1)

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.1,random_state=42)
X_combined = pd.concat([X1, y.astype(float)], axis=1)
print(X_combined.corr())
sns.heatmap(X_combined.corr(), annot=True, cmap="coolwarm")
#plt.show()

forest = RandomForestRegressor(random_state=42)
forest.fit(X_train,y_train)
predictions_forest = forest.predict(X_test)
score_forest = r2_score(y_test,predictions_forest)
absolute_forest = mean_absolute_error(y_test,predictions_forest,multioutput='raw_values')

reg = LinearRegression()
reg.fit(X_train,y_train)
predictionsreg = reg.predict(X_test)
r2_reg = r2_score(y_test,predictionsreg)
absolute_reg = mean_absolute_error(y_test,predictionsreg,multioutput='raw_values')

poly = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
poly.fit(X_train, y_train)
predictions_poly = poly.predict(X_test)
score_poly = r2_score(y_test,predictions_poly)
absolute_poly = mean_absolute_error(y_test,predictions_poly,multioutput='raw_values')
print(score_poly)

tree = DecisionTreeRegressor()
#cross_val_score(tree,X,y,cv=30)
tree.fit(X_train, y_train)
predictions_tree = tree.predict(X_test)
r2_tree = r2_score(y_test,predictions_tree)
absolute_tree = mean_absolute_error(y_test,predictions_tree,multioutput='raw_values')

scores = [['Regresja liniowa',r2_reg,absolute_reg],['Regresja wielomianowa',score_poly,absolute_poly],['Drzewa decyzyjne',r2_tree,absolute_tree],
          ['Las losowy',score_forest,absolute_forest]]
df_scores = pd.DataFrame(scores)
df_scores.columns = 'Algorytm','Współczynnik determinacji','Średni błąd bezwzględny'
print(df_scores)