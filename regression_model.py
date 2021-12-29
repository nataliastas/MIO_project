import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
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
X = data.drop(columns=column_names)
X = imputer.fit_transform(X)
y1 = data[['16-B']]# X7-6% imputer2 median
y2 = data[['16-P']]# X3-13% imputer2 most_frequent
y3 = data[['11-B']]# X3-33% imputer
y4 = data[['11-P']]# X3-16% imputer
y5 = data[['24-B']]# X2-28% imputer2 median
y6 = data[['24-P']]#??
y7 = data[['36-B']]#??
y8 = data[['36-P']]# X3-20% imputer1
y9 = data[['31-B']]# X2-13% imputer2 median
y10 = data[['31-P']]# X3-6% imputer1
y11 = data[['44-B']]# X4-5& imputer2 mean
y12 = data[['44-P']] # X2 -35% imputer 2 most frequent


model = sm.OLS(y3,X).fit()
print(model.summary())