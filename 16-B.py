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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

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
imputer  = KNNImputer()
imputer1 = IterativeImputer()
imputer2 = SimpleImputer(strategy='median')
#print(data.head())

column_names = ['16-B','16-P','11-B','11-P','24-B','24-P','36-B','36-P','31-B','31-P','44-B','44-P']

y1 = data[['16-B']]# X7-6% imputer2 median
X10 = data[['PI - 11', 'PPD - 36', 'PPD - 31 B', 'PI - 36', 'PI - 31', 'PI - 16', 'Interleukina ??? 31P', 'Interleukina ??? 16P', 'PPD - 44',
             'Interleukina ??? 36B', 'Interleukina ??? 24B', 'Interleukina ??? 11P', 'szyna', 'ograniczone otwieranie', 'bol miesni', 'TWI - 11 suma',
            'PPD - 44 P', 'PPD - 36 B', 'PPD - 24 P', 'PPD - 16', 'PPD - 11 P', 'PPD - 11', 'Interleukina ??? 16B', 'GI - 44', 'API', 'wiek',
            'przerost zwaczy', 'TWI - 24 suma', 'PPD - 24 B', 'GI - 24', 'ubytki klinowe', 'sztywnosc', 'TWI - 36 suma', 'TWI - 16 suma']]

X10 = imputer2.fit_transform(X10)
#print(X2)
X_train,X_test,y_train,y_test=train_test_split(X10,y1,test_size=0.1,random_state=42)

#sns.heatmap(X_combined.corr(), annot=True, cmap="coolwarm")
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

tree = DecisionTreeRegressor()
#cross_val_score(tree,X,y,cv=30)
tree.fit(X_train, y_train)
predictions_tree = tree.predict(X_test)
r2_tree = r2_score(y_test,predictions_tree)
absolute_tree = mean_absolute_error(y_test,predictions_tree,multioutput='raw_values')

scores = [['Regresja liniowa',r2_reg,absolute_reg],['Regresja wielomianowa',score_poly,absolute_poly],['Drzewa decyzyjne',r2_tree,absolute_tree],
          ['Las losowy',score_forest,absolute_forest]]
df_scores = pd.DataFrame(scores)
df_scores.columns = 'Algorytm','Wsp????czynnik determinacji','??redni b????d bezwzgl??dny'
print(df_scores)
