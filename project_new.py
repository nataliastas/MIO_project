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
imputer3 = SimpleImputer(strategy='mean')
imputer4 = SimpleImputer(strategy='most_frequent')
imputers = [imputer,imputer1,imputer2,imputer3,imputer4]

column_names = ['16-B','16-P','11-B','11-P','24-B','24-P','36-B','36-P','31-B','31-P','44-B','44-P','Unnamed: 0']
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

X = data.drop(columns=column_names)

#X = X.drop(0,axis=1,inplace=True)
feature_name = X.columns.tolist()
for i in range(0,5):
    X = imputers[i].fit_transform(X)
    X = pd.DataFrame(X,columns=feature_name)
    #print(feature_name)
    num_feats = 9
    #korelacja Pearsona
    #def cor_selector(X, y,num_feats):
    #cor_list = []
    #feature_name = X.columns.tolist()
        # calculate the correlation with y for each feature
    #for i in X.columns.tolist():
    #    cor = np.corrcoef(X[i], y)[0, 1]
    #    cor_list.append(cor)
        # replace NaN with 0
    #cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
    #cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
    #cor_support = [True if i in cor_feature else False for i in feature_name]
        #return cor_support, cor_feature
    #cor_support, cor_feature = cor_selector(X, y,num_feats)

    #chi-kwadrat
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import MinMaxScaler
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2)
    chi_selector.fit(X_norm, y1)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()

    #f_regression
    from sklearn.feature_selection import f_regression
    regression_selector = SelectKBest(f_regression)
    regression_selector.fit(X_norm, y1)
    regression_support = regression_selector.get_support()
    regression_feature = X.loc[:,regression_support].columns.tolist()

    #RFE
    from sklearn.feature_selection import RFE
    rfe_selector = RFE(estimator=LinearRegression(), step=10, verbose=5)
    rfe_selector.fit(X_norm, y1)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()

    rfe2_selector = RFE(estimator=RandomForestRegressor(), step=10, verbose=5)
    rfe2_selector.fit(X_norm, y1)
    rfe2_support = rfe2_selector.get_support()
    rfe2_feature = X.loc[:,rfe2_support].columns.tolist()

#select from model
    from sklearn.feature_selection import SelectFromModel
    embeded_lr_selector = SelectFromModel(LinearRegression())
    embeded_lr_selector.fit(X_norm, y1)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()

#random forest select from model
    embeded_rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=100))
    embeded_rf_selector.fit(X, y1)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()

#display
# put all selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_name,  'Chi-2':chi_support, 'F_regression':regression_support,'RFE':rfe_support, 'RFE2':rfe2_support,
                                     'SFM regression':embeded_lr_support,'SFM Random Forest':embeded_rf_support})
# count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    feature_selection_df.head(num_feats)
    new_df = feature_selection_df[feature_selection_df['RFE']==True]
    X_names = new_df['Feature'].tolist()
    print(X_names)
    X10 = data[X_names]

    X10 = imputers[i].fit_transform(X10)
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
    df_scores.columns = 'Algorytm','Współczynnik determinacji','Średni błąd bezwzględny'
    print(df_scores)
