import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,f1_score
import pickle


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC

def model_building(X, y, test, model, params = None, k = 1) :
    
    if params == None :
        model.fit(X, y)
        
        # return fitted model & train-test predictions
        return (model, model.predict(X), model.predict(test))
    
    else :
        model_cv = GridSearchCV(model, param_grid = params, cv = k)
        model_cv.fit(X, y)
        model = model_cv
        
        # return and extra object for all cross validation operations
        return (model_cv, model, model.predict(X), model.predict(test))
    
def model_evaluation(y_train, pred_train, y_test, pred_test) :
    
    print('''
            +--------------------------------------+
            | CLASSIFICATION REPORT FOR TRAIN DATA |
            +--------------------------------------+''')
    print(classification_report(y_train, pred_train))
    print(confusion_matrix(y_train, pred_train))
    print("F1 Score: ",f1_score(y_train, pred_train,average="macro"))
    
    print('''
            +--------------------------------------+
            | CLASSIFICATION REPORT FOR TEST DATA  |
            +--------------------------------------+''')
    print(classification_report(y_test, pred_test))
    print(confusion_matrix(y_test, pred_test))
    print("F1 Score: ",f1_score(y_test, pred_test,average="macro"))


# data_raw = pd.read_excel("../Datasets/E Commerce Dataset.xlsx",sheet_name="E Comm")
data_raw = pd.read_excel("D:/Academics & Learning/MSc in Artificial Intelligence/Q7/Master's Dissertation Part 1/Datasets/E Commerce Dataset.xlsx",sheet_name="E Comm")

col_list = ["Tenure","WarehouseToHome","HourSpendOnApp","OrderAmountHikeFromlastYear","CouponUsed","OrderCount","DaySinceLastOrder"]
for col in col_list:
    values_arr = data_raw[col].dropna()
    values_arr = values_arr.astype(float)
    mean_value = values_arr.mean()
    data_raw[col].fillna(mean_value, inplace=True)
    data_raw[col] = data_raw[col].astype(float)

data_processed = data_raw.copy()
data_processed.drop("CustomerID", axis=1, inplace=True)

data_processed["Churn"].replace({0:"No",1: "Yes"}, inplace=True)
data_processed["Complain"].replace({0:"No",1: "Yes"}, inplace=True)

X = data_processed.drop("Churn",axis=1)
y = data_processed["Churn"]

y.replace({"Yes":1,"No":0},inplace=True)

X_enc = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_enc,y, test_size=0.1,random_state=111,stratify=y)

model_list =[GaussianNB()]
for model in model_list:

    print("\n***** Performing for {0} *****".format(model))
    model, pred_train, pred_test = model_building(X_train, y_train,X_test,model,params = None)
    model_evaluation(y_train, pred_train, y_test, pred_test)

pickle.dump(model, open("D:/Academics & Learning/MSc in Artificial Intelligence/Q7/Master's Dissertation Part 1/Models/model.pkl",'wb'))

print("Model file executed successfully")