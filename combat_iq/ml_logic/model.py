import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

import catboost
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from combat_iq.ml_logic.preprocessors import preprocessed_df


def train_pipline(data=None):
    """
    crate, train and save binary classification model on fights pandas dataframe
    parameters : pandas dataframe from Kaggle UFC stats competition
    """

    if not data:
        data = pd.read_csv('../../raw_data/data.csv')    
    
    X = data.drop(['Winner'], axis=1)
    y= data.Winner
    X = X.replace('NaN', np.nan)
    # Replace non-Red values in Winner-column for 2-class-classification
    y = y.apply(lambda x: 1 if x=='Red' else 0)
    
    columns_to_drop = X.isna().sum().sort_values()[-109:].index.to_list() +['date', 'location','date', 'title_bout', 'weight_class']
    X = X.drop(columns=columns_to_drop, axis=1)

    categorical_column_names = X.select_dtypes(include=['object']).columns.to_list()
    categorical_indices = [i for i, v in enumerate(X.columns) if v in categorical_column_names]
    
    #preprocessing
    num_preproc = Pipeline([
        ("to_log", FunctionTransformer(np.log)),
        ("num_imputer", SimpleImputer(strategy = "median")),
        ("scaler", RobustScaler())])
    cat_preproc = Pipeline([
        ("cat_imputer", SimpleImputer(strategy = "constant", fill_value="Unknown")) ])
    bool_preproc = Pipeline([
        ("bool_imputer", SimpleImputer(strategy = "most_frequent")),
        ("to_str", FunctionTransformer(str))])
    
    preproc = ColumnTransformer([
        ("num_tr", num_preproc, make_column_selector(dtype_include = ["float64", "int64"])),
        ("cat_tr", cat_preproc, make_column_selector(dtype_include = ["object"])),
        ("bool_tr", bool_preproc, make_column_selector(dtype_include = ["bool"]))
    ], remainder="passthrough")
    
    cv = StratifiedKFold(n_splits = 5)
    model3 = catboost.CatBoostClassifier(n_estimators=2500, depth=5, learning_rate=0.04,silent=True,
                                             cat_features=categorical_indices,
                                             eval_metric='AUC')
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, 
                                                       train_size=0.8, 
                                                       random_state=42, stratify = y)
    X_train = pd.DataFrame(data=X_train, columns=X.columns)
    X_test = pd.DataFrame(data=X_test, columns=X.columns)
    
    model3_pipe = Pipeline([
        ("preproc", preproc),
        ("model3_classifier", model3)])
    
    model3_pipe_mean_accuracy = cross_val_score(model3_pipe, X_train, y=y_train, scoring='accuracy', cv=cv).mean()
    print(f"Cross-validation mean accuracy for model3_pipe is {model3_pipe_mean_accuracy}")
    model3_pipe.fit(X_train,y_train)
    print(f"model3_pipe trained on all rows")   
    y_pred = model3_pipe.predict(X_test)
    model3_pipe_test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for model3_pipe is {model3_pipe_test_accuracy}")
       
    # Save pipeline   
    with open(f'../models/of_model3_acc{round(model3_pipe_test_accuracy*100000)}.pkl', 'wb') as file:
        pickle.dump(model3_pipe, file)
    print(f"model3_pipe is successfully saved as 'of_model3_acc{round(model3_pipe_test_accuracy*100000)}.pkl'")
    
    return model3_pipe
    
    
    
def predict(red_fighter: str, blue_fighter: str):
    """
    return the model prediction : if the fighter in RED corner will win the fight or not
    parameters : the user selects 2 fighters corresponding to red and blue corners
    """

    # importing the data then preprocessing to get specific data corresponding to the red and blue fighters
    fight_data = preprocessed_df(red_fighter, blue_fighter)

    # importing the model
    with open('models/of_model3_acc079468.pkl', 'rb') as file:
        model = pickle.load(file)

    # predicting the outcome of the fight
    prediction = model.predict(fight_data)[0] # 1 for #Red wins', 0 for "No Red wins" 
    win_rate = np.max(model.predict_proba(fight_data))

    # returning the outcome to the user through the API
    return {'fight_outcome' : f'{red_fighter}{[" will not ", ""][prediction]} win{["", "s"][prediction]}',
                'confidence_rate': round(win_rate, 3)
                    }

