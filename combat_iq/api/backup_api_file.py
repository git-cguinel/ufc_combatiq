from fastapi import FastAPI
import pickle
import pandas as pd
from combat_iq.ml_logic.preprocessors import preprocessed_df

app = FastAPI()

@app.get("/")
def root():
    return {'setup': "I'm on it dudes !"}

@app.get("/predict")
def predict(red_fighter: str, blue_fighter: str):
    """
    return the model prediction : the name of the fighter who wins the fight
    parameters : the user selects 2 fighters corresponding to red and blue corners
    """

    # importing the data then preprocessing to get specific data corresponding to the red and blue fighters
    fight_data = preprocessed_df(red_fighter, blue_fighter)

    # importing the model
    with open('models/simple_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # predicting the outcome of the fight
    prediction = model.predict(fight_data)[0]
    win_rate = {
        'Blue': f'{model.predict_proba(fight_data)[0][0] * 100:.1f} %',
        'Red': f'{model.predict_proba(fight_data)[0][2] * 100:.1f} %', # the index [1] stands for 'Draw'
    }

    # returning the outcome to the user through the API
    if prediction == "Red":
        return {'fight_outcome' : red_fighter,
                'win_rate': win_rate['Red']
                    }
    else:
        return {'fight_outcome': blue_fighter,
                'win_rate': win_rate['Blue'],
                }
