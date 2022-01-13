import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#def perform_processing(
#        input_data: pd.DataFrame
#) -> pd.DataFrame:
input_data = pd.read_csv('data_project.csv', header=0, sep='\t')
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(input_data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data

    # for the simplest approach generate a random DataFrame with proper column names and size

#my cod
def preprocess_data():
    enc = LabelEncoder()
    input_data['zgrzytanie'] = enc.fit_transform(input_data['zgrzytanie'])
    input_data['zaciskanie'] = enc.fit_transform(input_data['zaciskanie'])
    input_data['sztywnosc'] = enc.fit_transform(input_data['sztywnosc'])
    input_data['ograniczone otwieranie'] = enc.fit_transform(input_data['ograniczone otwieranie'])
    input_data['bol miesni'] = enc.fit_transform(input_data['bol miesni'])
    input_data['przygryzanie'] = enc.fit_transform(input_data['przygryzanie'])
    input_data['cwiczenia'] = enc.fit_transform(input_data['cwiczenia'])
    input_data['szyna'] = enc.fit_transform(input_data['szyna'])
    input_data['starcie-przednie'] = enc.fit_transform(input_data['starcie-przednie'])
    input_data['starcie-boczne'] = enc.fit_transform(input_data['starcie-boczne'])
    input_data['ubytki klinowe'] = enc.fit_transform(input_data['ubytki klinowe'])
    input_data['pekniecia szkliwa'] = enc.fit_transform(input_data['pekniecia szkliwa'])
    input_data['impresje jezyka'] = enc.fit_transform(input_data['impresje jezyka'])
    input_data['linea alba'] = enc.fit_transform(input_data['linea alba'])
    input_data['przerost zwaczy'] = enc.fit_transform(input_data['przerost zwaczy'])
    input_data['tkliwosc miesni'] = enc.fit_transform(input_data['tkliwosc miesni'])
    input_data['plec'] = enc.fit_transform(input_data['plec'])
    input_data['API'] = input_data['API'].str[:-1]
    input_data['SBI'] = input_data['SBI'].str[:-1]

def preprocess2(imputer, X_columns):
    X = input_data[X_columns]
    X = imputer.fit_transform(X)
    return X
#example
column_names = ['16-B', '16-P', '11-B', '11-P', '24-B', '24-P', '36-B', '36-P', '31-B', '31-P', '44-B', '44-P']
predicted_data = pd.DataFrame(
    np.random.randint(low=0, high=100, size=(len(input_data.index), len(column_names))),
    columns=column_names
)
#end example
preprocess_data()
imputer = KNNImputer()
imputer1 = IterativeImputer()
imputer2 = SimpleImputer(strategy='median')
imputer3 = SimpleImputer(strategy='mean')
imputer4 = SimpleImputer(strategy='most_frequent')

y1 = input_data[['16-B']]  # imputer2
X1_columns = ['PI - 11', 'PPD - 36', 'PPD - 31 B', 'PI - 36', 'PI - 31', 'PI - 16', 'Interleukina – 31P','Interleukina – 16P', 'PPD - 44',
            'Interleukina – 36B', 'Interleukina – 24B', 'Interleukina – 11P', 'szyna', 'ograniczone otwieranie','bol miesni', 'TWI - 11 suma',
            'PPD - 44 P', 'PPD - 36 B', 'PPD - 24 P', 'PPD - 16', 'PPD - 11 P', 'PPD - 11','Interleukina – 16B', 'GI - 44', 'API', 'wiek',
            'przerost zwaczy', 'TWI - 24 suma', 'PPD - 24 B', 'GI - 24', 'ubytki klinowe', 'sztywnosc','TWI - 36 suma', 'TWI - 16 suma']
X1 = preprocess2(imputer2,X1_columns)
y2 = input_data[['16-P']]  # imputer4
X2_columns = ['sztywnosc', 'PPD - 36', 'PPD - 31', 'PPD - 24', 'PPD - 11', 'Interleukina – 36P', 'Interleukina – 24P','GI - 24', 'PPD - 44',
            'PPD - 24 P', 'PPD - 24 B', 'PPD - 16 P', 'PPD - 16', 'PI - 24', 'Interleukina – 44B','Interleukina – 24B', 'Interleukina – 16P',
            'Interleukina – 16B', 'GI - 16', 'szyna', 'linea alba', 'TWI - 24 suma', 'PI - 16','Interleukina – 31B', 'GI - 31', 'GI - 11', 'wiek',
            'przerost zwaczy', 'ograniczone otwieranie', 'TWI - 31 suma', 'TWI - 16 suma', 'PPD - 31 P','starcie-przednie', 'pekniecia szkliwa']
X2 = preprocess2(imputer4,X2_columns)
y3 = input_data[['11-B']]  # imputer
X3_columns = ['PI - 16', 'PI - 11', 'PI - 24', 'PI - 36', 'PI - 31', 'PI - 44', 'GI - 16', 'GI - 11', 'GI - 24', 'GI - 36','GI - 31', 'GI - 44',
            'PPD - 16', 'PPD - 11', 'PPD - 24', 'PPD - 36', 'PPD - 31', 'PPD - 44', 'TWI - 11 suma', 'TWI - 16 suma','TWI - 24 suma',
            'TWI - 36 suma','TWI - 31 suma', 'TWI - 44 suma', 'Interleukina – 11B', 'Interleukina – 11P', 'Interleukina – 16B',
            'Interleukina – 16P', 'Interleukina – 24B','Interleukina – 24P', 'Interleukina – 31B', 'Interleukina – 31P', 'Interleukina – 36B',
            'Interleukina – 36P','Interleukina – 44B','Interleukina – 44P','zgrzytanie', 'zaciskanie', 'sztywnosc', 'ograniczone otwieranie',
            'bol miesni','przygryzanie', 'cwiczenia', 'szyna','starcie-przednie', 'starcie-boczne', 'ubytki klinowe','impresje jezyka',
            'linea alba', 'przerost zwaczy', 'tkliwosc miesni']
X3 = preprocess2(imputer,X3_columns)
y4 = input_data[['11-P']]  # imputer
X4_columns = ['PI - 36', 'PI - 24', 'PI - 11', 'GI - 31', 'PPD - 44 P', 'PPD - 36 B', 'PPD - 16 P', 'GI - 24', 'PPD - 36','PPD - 24 P',
            'PPD - 24', 'PPD - 16', 'PI - 31', 'Interleukina – 36B', 'Interleukina – 24B', 'Interleukina – 16P','Interleukina – 11P',
            'Interleukina – 11B', 'GI - 16', 'API', 'tkliwosc miesni', 'impresje jezyka', 'PPD - 44', 'PPD - 36 P','PPD - 11 P', 'wiek',
            'ograniczone otwieranie', 'cwiczenia', 'TWI - 44 suma', 'TWI - 36 suma', 'PPD - 31 P', 'PPD - 11 B', 'GI - 36','linea alba']
X4 = preprocess2( imputer, X4_columns)
y5 = input_data[['24-B']]  # imputer4/1
X5_columns = ['GI - 31', 'PPD - 44', 'PPD - 31 B', 'PPD - 11', 'PI - 44', 'PI - 24', 'Interleukina – 16B', 'GI - 24', 'PPD - 36 P', 'PPD - 36',
            'PPD - 16 B', 'PI - 31', 'Interleukina – 44B', 'Interleukina – 31P', 'Interleukina – 24P', 'Interleukina – 16P', 'Interleukina – 11B',
            'GI - 16', 'PPD - 44 P', 'PPD - 44 B', 'PPD - 16', 'PI - 11', 'PPD - 31', 'PPD - 24', 'PI - 36', 'Interleukina – 44P',
            'Interleukina – 36P', 'Interleukina – 36B', 'Interleukina – 31B', 'Interleukina – 24B', 'Interleukina – 11P', 'GI - 44', 'GI - 11', 'API']
X5 = preprocess2( imputer1, X5_columns)
y6 = input_data[['24-P']]  # imputer
X6_columns = ['PI - 36', 'PI - 24', 'PI - 16', 'PPD - 24', 'GI - 31', 'GI - 24', 'TWI - 16 suma', 'PPD - 44', 'PPD - 36 P','PPD - 36',
            'PPD - 16 P','Interleukina – 44B', 'Interleukina – 36P', 'Interleukina – 36B', 'Interleukina – 11P', 'tkliwosc miesni',
            'TWI - 11 suma', 'PPD - 31 B','PI - 11', 'Interleukina – 24B', 'GI - 36', 'zaciskanie', 'szyna', 'impresje jezyka', 'TWI - 36 suma',
            'TWI - 31 suma', 'TWI - 24 suma','PPD - 16 B', 'starcie-przednie', 'starcie-boczne', 'plec', 'TWI - 44 suma', 'PPD - 36 B',
            'PPD - 11 B']
X6 = preprocess2( imputer, X6_columns)
y7 = input_data[['36-B']]  # imputer
X7_columns = ['wiek', 'PPD - 11', 'Interleukina – 44B', 'Interleukina – 36B', 'Interleukina – 16B', 'GI - 31', 'GI - 11','PPD - 44 B',
            'PPD - 44','PPD - 24', 'PI - 36', 'PI - 16', 'PI - 11', 'Interleukina – 36P', 'Interleukina – 24B', 'Interleukina – 11P',
            'Interleukina – 11B','tkliwosc miesni', 'szyna', 'linea alba', 'impresje jezyka', 'PPD - 36 B', 'PPD - 36', 'PPD - 16 B',
            'PPD - 11 P', 'PI - 44','Interleukina – 31B', 'GI - 44', 'TWI - 36 suma', 'TWI - 16 suma', 'PPD - 11 B', 'GI - 36', 'TWI - 44 suma',
            'PPD - 16 P']
X7 = preprocess2( imputer, X7_columns)
y8 = input_data[['36-P']]  # imputer
X8_columns = ['PPD - 44 P', 'PPD - 31', 'PI - 11', 'GI - 36', 'PPD - 44', 'PPD - 36', 'PPD - 24 B', 'PPD - 24', 'PPD - 16 P',
            'PPD - 11 P', 'PI - 36','PI - 31', 'PI - 16', 'Interleukina – 36P', 'Interleukina – 24B', 'Interleukina – 16P', 'Interleukina – 11P',
            'Interleukina – 11B','ograniczone otwieranie', 'cwiczenia', 'TWI - 36 suma', 'TWI - 31 suma', 'TWI - 11 suma', 'PPD - 36 B',
            'Interleukina – 24P', 'GI - 24','wiek', 'sztywnosc', 'przygryzanie', 'TWI - 44 suma', 'PPD - 44 B', 'PPD - 24 P', 'PPD - 16 B',
            'szyna']
X8 = preprocess2(imputer, X8_columns)
y9 = input_data[['31-B']]  # imputer4
X9_columns = ['PI - 24', 'PI - 16', 'GI - 31', 'PPD - 36 B', 'PPD - 36', 'PPD - 31 B', 'PI - 36', 'GI - 24', 'PPD - 31',
            'PPD - 24 P', 'PPD - 16 B','PI - 44', 'PI - 31', 'Interleukina – 24B', 'Interleukina – 16P', 'Interleukina – 11P',
            'Interleukina – 11B','GI - 44', 'GI - 11','tkliwosc miesni', 'szyna', 'impresje jezyka', 'bol miesni', 'TWI - 11 suma',
            'zaciskanie', 'wiek','ograniczone otwieranie','TWI - 36 suma', 'TWI - 24 suma', 'TWI - 16 suma', 'PPD - 11 P']
X9 = preprocess2( imputer4, X9_columns)
y10 = input_data[['31-P']]  # imputer
X10_columns = ['PPD - 36 P', 'PPD - 24 P', 'PPD - 16', 'PPD - 11 P', 'PI - 24', 'Interleukina – 44P', 'Interleukina – 24P',
            'ubytki klinowe','TWI - 24 suma', 'PPD - 44 P', 'PPD - 36', 'PPD - 31', 'PI - 31', 'Interleukina – 36B', 'Interleukina – 16P',
            'Interleukina – 16B','Interleukina – 11P', 'wiek', 'tkliwosc miesni', 'szyna', 'ograniczone otwieranie', 'linea alba',
            'impresje jezyka', 'PPD - 11','Interleukina – 31P', 'GI - 31', 'cwiczenia', 'bol miesni', 'TWI - 44 suma', 'TWI - 16 suma',
            'TWI - 11 suma','PPD - 31 B', 'PPD - 24 B','PPD - 11 B']
X10 = preprocess2( imputer, X10_columns)
y11 = input_data[['44-B']]  # imputer
X11_columns = ['wiek', 'PPD - 44', 'PPD - 24 P', 'PPD - 24 B', 'PPD - 16', 'PPD - 11', 'TWI - 24 suma', 'PPD - 36 P','PPD - 16 B', 'PI - 44',
            'PI - 36', 'Interleukina – 44P', 'Interleukina – 36P', 'Interleukina – 36B', 'Interleukina – 31B','Interleukina – 24B', 'GI - 31',
            'linea alba', 'impresje jezyka', 'cwiczenia', 'PPD - 31 P', 'PI - 24', 'PI - 11', 'Interleukina – 31P','Interleukina – 16B',
            'Interleukina – 11P', 'GI - 36', 'ubytki klinowe', 'tkliwosc miesni', 'szyna', 'plec','pekniecia szkliwa', 'TWI - 44 suma',
            'TWI - 16 suma']
X11 = preprocess2( imputer, X11_columns)
y12 = input_data[['44-P']]  # imputer4
X12_columns = ['PI - 16', 'GI - 31', 'PPD - 44', 'PPD - 36 B', 'PPD - 11', 'wiek', 'PPD - 44 P', 'PPD - 36', 'PPD - 31','PPD - 16 P',
            'PPD - 16','PI - 36', 'PI - 31', 'PI - 24', 'Interleukina – 36P', 'Interleukina – 24B', 'Interleukina – 11P','tkliwosc miesni',
            'szyna','ograniczone otwieranie', 'TWI - 36 suma', 'TWI - 24 suma', 'PPD - 44 B', 'PPD - 31 B', 'Interleukina – 44B',
            'Interleukina – 31P','Interleukina – 24P', 'linea alba', 'TWI - 11 suma', 'PPD - 36 P', 'API', 'PPD - 24 P', 'PPD - 11 B','GI - 24']
X12 = preprocess2( imputer4, X12_columns)


def data_train(input_data,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train,X_test,y_train,y_test
X1_train,X1_test,y1_train,y1_test = data_train(input_data, X1, y1)
X2_train,X2_test,y2_train,y2_test = data_train(input_data, X2, y2)
X3_train,X3_test,y3_train,y3_test = data_train(input_data, X3, y3)
X4_train,X4_test,y4_train,y4_test = data_train(input_data, X4, y4)
X5_train,X5_test,y5_train,y5_test = data_train(input_data, X5, y5)
X6_train,X6_test,y6_train,y6_test = data_train(input_data, X6, y6)
X7_train,X7_test,y7_train,y7_test = data_train(input_data, X7, y7)
X8_train,X8_test,y8_train,y8_test = data_train(input_data, X8, y8)
X9_train,X9_test,y9_train,y9_test = data_train(input_data, X9, y9)
X10_train,X10_test,y10_train,y10_test = data_train(input_data, X10, y10)
X11_train,X11_test,y11_train,y11_test = data_train(input_data, X11, y11)
X12_train,X12_test,y12_train,y12_test = data_train(input_data, X12, y12)

def predict(X_train,y_train,X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions
prediction1 = predict(X1_train, y1_train, X1_test).flatten()
prediction2 = predict(X2_train, y2_train, X2_test).flatten()
prediction3 = predict(X3_train, y3_train, X3_test).flatten()
prediction4 = predict(X4_train, y4_train, X4_test).flatten()
prediction5 = predict(X5_train, y5_train, X5_test).flatten()
prediction6 = predict(X6_train, y6_train, X6_test).flatten()
prediction7 = predict(X7_train, y7_train, X7_test).flatten()
prediction8 = predict(X8_train, y8_train, X8_test).flatten()
prediction9 = predict(X9_train, y9_train, X9_test).flatten()
prediction10 = predict(X10_train, y10_train, X10_test).flatten()
prediction11 = predict(X11_train, y11_train, X11_test).flatten()
prediction12 = predict(X12_train, y12_train, X12_test).flatten()

predicted = [prediction1,prediction2,prediction3,prediction4,prediction5,prediction6,prediction7,prediction8,prediction9,prediction10,
             prediction11,prediction12]

column_names = {'0':'16-B', '1':'16-P', '2':'11-B','3': '11-P','4': '24-P', '5':'36-B', '6':'36-P', '7':'31-B','8': '31-P','9': '44-B'}
predicted_data = pd.DataFrame(predicted).T
predicted_data = predicted_data.rename(columns=column_names)
print(predicted_data)
