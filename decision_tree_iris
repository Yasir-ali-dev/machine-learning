
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score


#%% loading Data
X,y=load_iris(return_X_y=True)

#%% Splitting data into Training and Testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=32,)

#%% Model Fitting 
#set the 'criterion' to 'entropy', which sets the measure for splitting the attribute to information gain.
# criterion can be "gini" 
decision_tree_classifier=DecisionTreeClassifier(criterion="entropy")
decision_tree_classifier.fit(X_train, y_train)
y_pred=decision_tree_classifier.predict(X_test)

#%% Evaluation 
print(confusion_matrix(y_test, y_pred))
print("\nf1_score : ", f1_score(y_test, y_pred,average="weighted"))
print("\nAccuracy Score : ",accuracy_score(y_test, y_pred))

