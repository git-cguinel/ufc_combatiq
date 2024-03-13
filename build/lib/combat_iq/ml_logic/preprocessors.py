import pandas as pd
from ml_logic.data import get_data


def preprocessed_df(red_fighter, blue_fighter):

    all_fighters = get_data()

    X_blue = all_fighters[all_fighters['fighter'] == blue_fighter]
    X_blue.columns = ["B_"+col for col in X_blue.columns]

    X_red = all_fighters[all_fighters['fighter'] == red_fighter]
    X_red.columns = ["R_"+col for col in X_red.columns]

    new_column_names = pd.concat([X_red, X_blue]).columns
    data = list(X_red.iloc[0]) + list(X_blue.iloc[0])

    fight_df = pd.DataFrame(data).T
    fight_df.columns = new_column_names

    fight_df = fight_df[['B_total_rounds_fought', 'B_total_title_bouts', 'B_current_win_streak',
       'B_current_lose_streak', 'B_longest_win_streak', 'B_wins', 'B_losses',
       'B_draw', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split',
       'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission',
       'B_win_by_TKO_Doctor_Stoppage', 'R_total_rounds_fought',
       'R_total_title_bouts', 'R_current_win_streak', 'R_current_lose_streak',
       'R_longest_win_streak', 'R_wins', 'R_losses', 'R_draw',
       'R_win_by_Decision_Majority', 'R_win_by_Decision_Split',
       'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission',
       'R_win_by_TKO_Doctor_Stoppage']]

    return fight_df
