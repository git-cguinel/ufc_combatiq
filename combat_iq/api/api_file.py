from fastapi import FastAPI
import pickle
import pandas as pd
import umpy as np
from combat_iq.ml_logic.preprocessors import preprocessed_df

app = FastAPI()

@app.get("/")
def root():
    return {'setup': "I'm on it dudes !"}

@app.get("/predict")
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
