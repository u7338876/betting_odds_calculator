from predict_odds import match_odds 
from predict_odds import get_probability_array
import pandas as pd
import numpy as np
import argparse

def df_epl_24_25(url):
    # Read CSV, standardise team names, and remove newly promoted teams
    df = pd.read_csv(url)
    columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    df['HomeTeam'] = df['HomeTeam'].replace('Man United', 'Manchester United')
    df['AwayTeam'] = df['AwayTeam'].replace('Man United', 'Manchester United')
    df['HomeTeam'] = df['HomeTeam'].replace('Man City', 'Manchester City')
    df['AwayTeam'] = df['AwayTeam'].replace('Man City', 'Manchester City')
    df['HomeTeam'] = df['HomeTeam'].replace("Nott'm Forest", 'Nottingham Forest')
    df['AwayTeam'] = df['AwayTeam'].replace("Nott'm Forest", 'Nottingham Forest')
    df['HomeTeam'] = df['HomeTeam'].replace('Wolves', 'Wolverhampton Wanderers')
    df['AwayTeam'] = df['AwayTeam'].replace('Wolves', 'Wolverhampton Wanderers')
    df['HomeTeam'] = df['HomeTeam'].replace('Newcastle', 'Newcastle United')
    df['AwayTeam'] = df['AwayTeam'].replace('Newcastle', 'Newcastle United')
    df = df[~df['HomeTeam'].isin(['Leicester', 'Southampton', 'Ipswich'])]
    df = df[~df['AwayTeam'].isin(['Leicester', 'Southampton', 'Ipswich'])]
    df = df.reset_index(drop=True)
    df = df[columns]
    return df

def df_epl_23_24(url):
    # Read CSV, standardise team names, and remove newly promoted teams
    df = pd.read_csv(url)
    columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    df['HomeTeam'] = df['HomeTeam'].replace('Man United', 'Manchester United')
    df['AwayTeam'] = df['AwayTeam'].replace('Man United', 'Manchester United')
    df['HomeTeam'] = df['HomeTeam'].replace('Man City', 'Manchester City')
    df['AwayTeam'] = df['AwayTeam'].replace('Man City', 'Manchester City')
    df['HomeTeam'] = df['HomeTeam'].replace("Nott'm Forest", 'Nottingham Forest')
    df['AwayTeam'] = df['AwayTeam'].replace("Nott'm Forest", 'Nottingham Forest')
    df['HomeTeam'] = df['HomeTeam'].replace('Wolves', 'Wolverhampton Wanderers')
    df['AwayTeam'] = df['AwayTeam'].replace('Wolves', 'Wolverhampton Wanderers')
    df['HomeTeam'] = df['HomeTeam'].replace('Newcastle', 'Newcastle United')
    df['AwayTeam'] = df['AwayTeam'].replace('Newcastle', 'Newcastle United')
    df = df.reset_index(drop=True)
    df = df[columns]
    return df

def add_odds_to_df(df, df_attack, df_defence, df_home_advantage):
    # Calculate predicted odds for each match
    predicted_home_odds = []
    predicted_draw_odds = []
    predicted_away_odds = []
    for i in range(len(df)):
        odds = match_odds(get_probability_array(df['HomeTeam'][i], df['AwayTeam'][i], df_attack, df_defence, df_home_advantage))
        predicted_home_odds.append(odds[0])
        predicted_draw_odds.append(odds[1])
        predicted_away_odds.append(odds[2])
    df['Predicted_H'] = predicted_home_odds
    df['Predicted_D'] = predicted_draw_odds
    df['Predicted_A'] = predicted_away_odds
    return df

def kelly_criterion(actual_odds, predicted_odds, wallet):
    p = 1/predicted_odds
    q = 1 - p
    b = actual_odds-1
    return (b*p-q)*wallet/b

def count_winnings(df, wallet):
    winnings = 0
    for i in range(len(df)):
        # Bet if actual odds are more than predicted odds
        if df['B365H'][i] > df['Predicted_H'][i]:
            # Bet for home win
            toBet = kelly_criterion(df['B365H'][i], df['Predicted_H'][i], wallet)
            if df['FTR'][i] == 'H': # Home Win
                winnings += toBet * (df['B365H'][i] - 1)
            else:
                winnings -= toBet
        if df['B365D'][i] > df['Predicted_D'][i]:
            # Bet for draw
            toBet = kelly_criterion(df['B365D'][i], df['Predicted_D'][i], wallet)
            if df['FTR'][i] == 'D': # Draw
                winnings += toBet * (df['B365D'][i] - 1)
            else:
                winnings -= toBet
        if df['B365A'][i] > df['Predicted_A'][i]:
            # Bet for away win
            toBet = kelly_criterion(df['B365A'][i], df['Predicted_A'][i], wallet)
            if df['FTR'][i] == 'A': # Draw
                winnings += toBet * (df['B365A'][i] - 1)
            else:
                winnings -= toBet
    return winnings

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test how well model does on bets")
    parser.add_argument('league', type=str, help='Specify which league')
    parser.add_argument('season', type=str, help='Specify which season')
    parser.add_argument('wallet', type=float, help='Specify how much in your wallet')
    args = parser.parse_args()

    # Read data files
    attacking_scores_csv_path = './data/data_EPL/attacking_scores.csv'
    defending_scores_csv_path = './data/data_EPL/defending_scores.csv'
    home_advantage_csv_path = './data/data_EPL/home_advantage.csv'
    df_attack = pd.read_csv(attacking_scores_csv_path, index_col='team')
    df_defence = pd.read_csv(defending_scores_csv_path, index_col='team')
    df_home_advantage = pd.read_csv(home_advantage_csv_path, index_col='parameter')

    # Choose league and season
    df = pd.DataFrame()
    if args.league == 'EPL' and args.season == "24/25":
        df = df_epl_24_25('./data/betting_odds/EPL_24_25.csv')
    elif args.league == 'EPL' and args.season == "23/24":
        df = df_epl_23_24('./data/betting_odds/EPL_23_24.csv')
    else: 
        raise Exception('League and Season not found')

    # Count winnings
    df = add_odds_to_df(df, df_attack, df_defence, df_home_advantage)
    winnings = np.round(count_winnings(df, args.wallet),2)
    print(f'Potential winnings for {args.league} {args.season} season given ${args.wallet}: ${winnings}')
    

if __name__ == "__main__":
    main()