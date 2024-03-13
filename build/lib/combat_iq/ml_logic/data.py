import pandas as pd

def get_data():

    all_fighters = pd.read_csv("combat_iq/api/all_fighters.csv")

    return all_fighters
