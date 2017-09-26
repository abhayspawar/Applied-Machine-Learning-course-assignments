#Detailed explanation about the steps followed is written in the readme file.

def score_rent():
    import pandas as pd
    import io
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn import linear_model
    from sklearn.feature_selection import f_regression
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    c=pd.read_csv("https://ndownloader.figshare.com/files/7586326")
    train_data=c[c['uf17']<7000].reset_index(drop=True)
    #removing unncessary columns
    k=np.concatenate((range(30,65),[72,11,91,0,98,121,122,123,119,120],range(131,138),range(140,197)),axis=0)
    X=train_data.drop(train_data.columns[k], axis=1)
    #Removing all the continous features, so that it is easier to create dummies for categorical variables.
    #I haven't implemented pipeline for creating dummies. I saw that on splitting the data into test and train, 
    #both the datasets have all categories for a variable
    y=train_data['uf17']
    X.drop('uf12', axis=1, inplace=True) # 672 to 485 and 9999 to 0
    X.drop('uf13', axis=1, inplace=True) # 694 to 485 and 9999 to 0
    X.drop('uf14', axis=1, inplace=True) # 816 to 597 and 9999 to 0
    X.drop('uf15', axis=1, inplace=True) # 4587 to 3,200 and 9999 to 0
    X.drop('uf16', axis=1, inplace=True) #10388 to 7,800 
    X.drop('uf64', axis=1, inplace=True) #325 to 0696 9998=Not reported
    X.drop('rec54', axis=1, inplace=True) # deficiencies, 7 value is not reported, change to missing value
    X.drop('rec53', axis=1, inplace=True) # deficiencies, 7 value is not reported, change to missing value
    X.drop('sc27', axis=1, inplace=True) #not sure what this is. Dont use this

    #Creating dummies
    columns=X.columns
    X1=pd.get_dummies(X['boro'],prefix='boro')
    for i in range(1,len(columns)):
        X_curr=pd.get_dummies(X[columns[i]],prefix=columns[i])
        frames=[X1,X_curr]
        X1=pd.concat(frames,axis=1)
    #Adding the continous variables which improved CV to the dataset
    frames=[X1,train_data['sc150'],train_data['sc151'],train_data['uf16']]
    X1=pd.concat(frames,axis=1)
    
    #imputation
    for i in range(len(X1)):
        if X1['uf16'][i]==9999:
            X1['uf16'].set_value(0, 0)

    #Using LASSO to do feature selection
    X_train_val, X_test,y_train_val,y_test=train_test_split(X1,y,test_size=0.2,random_state=42)
    param_grid={'alpha':np.logspace(-1,1,5)}
    grid=GridSearchCV(linear_model.Lasso(),param_grid,cv=5)
    grid.fit(X_train_val,y_train_val)

    zeros=[]
    for i in range(len(grid.best_estimator_.coef_)):
        if grid.best_estimator_.coef_[i]==0:
            zeros.append(i)
    #Using features obtained from LASSO to build ridge regression model
    X1_drop=X1.drop(X1.columns[zeros], axis=1)

    Xdrop_train_val, Xdrop_test,y_train_val,y_test=train_test_split(X1_drop,y,test_size=0.2,random_state=42)

    param_grid={'alpha':np.logspace(-5,5,10)}
    grid=GridSearchCV(linear_model.Ridge(),param_grid,cv=5)
    grid.fit(Xdrop_train_val,y_train_val)
    return grid.best_estimator_.score(Xdrop_test,y_test)

def predict_rent():
    import pandas as pd
    import io
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn import linear_model
    from sklearn.feature_selection import f_regression
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    c=pd.read_csv("https://ndownloader.figshare.com/files/7586326")
    train_data=c[c['uf17']<7000].reset_index(drop=True)
    k=np.concatenate((range(30,65),[72,11,91,0,98,121,122,123,119,120],range(131,138),range(140,197)),axis=0)
    X=train_data.drop(train_data.columns[k], axis=1)
    y=train_data['uf17']
    X.drop('uf12', axis=1, inplace=True) # 672 to 485 and 9999 to 0
    X.drop('uf13', axis=1, inplace=True) # 694 to 485 and 9999 to 0
    X.drop('uf14', axis=1, inplace=True) # 816 to 597 and 9999 to 0
    X.drop('uf15', axis=1, inplace=True) # 4587 to 3,200 and 9999 to 0
    X.drop('uf16', axis=1, inplace=True) #10388 to 7,800 
    X.drop('uf64', axis=1, inplace=True) #325 to 0696 9998=Not reported
    X.drop('rec54', axis=1, inplace=True) # deficiencies, 7 value is not reported, change to missing value
    X.drop('rec53', axis=1, inplace=True) # deficiencies, 7 value is not reported, change to missing value
    X.drop('sc27', axis=1, inplace=True) #not sure what this is. Dont use this

    columns=X.columns
    X1=pd.get_dummies(X['boro'],prefix='boro')
    for i in range(1,len(columns)):
        X_curr=pd.get_dummies(X[columns[i]],prefix=columns[i])
        frames=[X1,X_curr]
        X1=pd.concat(frames,axis=1)

    frames=[X1,train_data['sc150'],train_data['sc151'],train_data['uf16']]
    X1=pd.concat(frames,axis=1)
    for i in range(len(X1)):
        if X1['uf16'][i]==9999:
            X1['uf16'].set_value(0, 0)

    X_train_val, X_test,y_train_val,y_test=train_test_split(X1,y,test_size=0.2,random_state=42)
    param_grid={'alpha':np.logspace(-1,1,5)}
    grid=GridSearchCV(linear_model.Lasso(),param_grid,cv=5)
    grid.fit(X_train_val,y_train_val)

    #grid.best_estimator_.coef_
    for i in range(len(grid.best_estimator_.coef_)):
        if grid.best_estimator_.coef_[i]==0:
            zeros.append(i)
    X1_drop=X1.drop(X1.columns[zeros], axis=1)

    Xdrop_train_val, Xdrop_test,y_train_val,y_test=train_test_split(X1_drop,y,test_size=0.2,random_state=42)

    param_grid={'alpha':np.logspace(-5,5,10)}
    grid=GridSearchCV(linear_model.Ridge(),param_grid,cv=5)
    grid.fit(Xdrop_train_val,y_train_val)
    y_pred=pd.Series(grid.best_estimator_.predict(Xdrop_test)).as_matrix()
    y_actual=y_test.as_matrix()
    return (Xdrop_test,y_actual,y_pred)