import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#%% importing Data
def ImportData (): 
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+
                               'databases/balance-scale/balance-scale.data',sep= ',',
                               header = None)
    X=balance_data.values[:,1:]
    y=balance_data.values[:,0]    
    return X,y

#%% Spliting data into Training and Testing
def SplitDataSet(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=45)    
    return X_train,X_test,y_train,y_test   


#%% Model 
def DecisionTreeClassifierModel(X_train,X_test,y_train):
    decision_tree_classifier=DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train,y_train)
    y_pred=decision_tree_classifier.predict(X_test)
    return y_pred

#%% Model Evaluation on 
def ModelEvaluation(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred) )
    print("\nAccuracy Score : ",accuracy_score(y_test, y_pred) )
    print("\nClassification Report : ",classification_report(y_test, y_pred) )


#%%% _main_
X,y=ImportData()
X_train,X_test,y_train,y_test=SplitDataSet(X, y)
y_pred=DecisionTreeClassifierModel(X_train, X_test, y_train)
ModelEvaluation(y_test, y_pred)


