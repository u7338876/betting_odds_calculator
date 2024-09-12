# Author: Ng Jun Kiat
# License: Creative Commons Attribution-NonCommercial (CC BY-NC)

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import argparse

# Code to retrieve data from URL

def data_from_url(url):
    # Send a GET request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    # Scrape data
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the script tag that contains the JSON data
    script_tag = soup.find('script', string=lambda s: s and 'JSON.parse' in s)
    raw_json_str = script_tag.string
    
    # Extract the part of the string that contains the JSON data
    start = raw_json_str.find("JSON.parse('") + len("JSON.parse('")
    end = raw_json_str.find("')")
    json_str = raw_json_str[start:end]
    
    # Decode the JSON string
    decoded_str = bytes(json_str, "utf-8").decode("unicode_escape")
    data = json.loads(decoded_str)

    return data

def data_to_df(data):
    # Initialize lists
    home_team = []
    away_team = []
    home_goals = []
    away_goals = []
    home_xg = []
    away_xg = []
    datetime = []
    home_win_prob = []
    draw_prob = []
    away_win_prob = []
    
    # Iterate through the data
    for match in data:
        home_team.append(match['h']['title'])
        away_team.append(match['a']['title'])
        home_goals.append(match['goals']['h'])
        away_goals.append(match['goals']['a'])
        home_xg.append(match['xG']['h'])
        away_xg.append(match['xG']['a'])
        datetime.append(match['datetime'])
        home_win_prob.append(match['forecast']['w'])
        draw_prob.append(match['forecast']['d'])
        away_win_prob.append(match['forecast']['l'])
    
    # Form DataFrame
    df = pd.DataFrame({
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'datetime': datetime,
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob
    })
    
    return df

def calculate_t(x, y, lambd, mil, p):
    if x == 0 and y == 0:
        return 1 - lambd*mil*p
    elif x == 0 and y == 1:
        return 1 + lambd*p
    elif x == 1 and y == 0:
        return 1 + mil*p
    elif x == 1 and y == 1:
        return 1 - p
    else: 
        return 1

def calculate_likelihood(parameters, df):
    # Retrieve list of teams in alphabetical order
    teams = np.sort(df['home_team'].unique())
    neg_log_likelihood = 0
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Count likelihood for each match 
    for i in range(len(df)):
        home_team = df.loc[i]['home_team']
        away_team = df.loc[i]['away_team']
        home_goals = int(df.loc[i]['home_goals'])
        away_goals = int(df.loc[i]['away_goals'])
        home_index = int(np.argwhere(teams == home_team)[0][0])
        away_index = int(np.argwhere(teams == away_team)[0][0])
        home_attack = parameters[home_index]
        home_defence = parameters[home_index + len(teams)]
        away_attack = parameters[away_index]
        away_defence = parameters[away_index + len(teams)]
        home_advantage = parameters[-2]
        p = parameters[-1]

        lambd = home_attack * away_defence * home_advantage
        mil = away_attack * home_defence

        t = calculate_t(home_goals, away_goals, lambd, mil, p)

        poisson_home = np.exp(-lambd) * np.power(lambd, home_goals)
        poisson_away = np.exp(-mil) * np.power(mil, away_goals)
        l = t * poisson_home * poisson_away

        if np.isnan(l) or l <= 0:
            l = epsilon

        neg_log_likelihood += -np.log(l)

    return neg_log_likelihood

# Define the optimization function
def objective_function(parameters, df):
    return calculate_likelihood(parameters, df)


def estimate_ad_score(league):
    url = ''
    if league == 'EPL':
        url = 'https://understat.com/league/EPL/2023'
    elif league == 'LaLiga':
        url = 'https://understat.com/league/La_liga/2023'
    elif league == 'Bundesliga':
        url = 'https://understat.com/league/Bundesliga/2023'
    elif league == 'Ligue1':
        url = 'https://understat.com/league/Ligue_1/2023'
    elif league == 'SerieA':
        url = 'https://understat.com/league/Serie_A/2023'
    else: 
        raise Exception('League not found.')
    data = data_from_url(url)
    df = data_to_df(data)

    # Initial guess
    num_teams = len(df['home_team'].unique())
    initial_guess = np.ones(2 * num_teams + 2) # Initialise all attacking and defending scores to 1
    initial_guess[-1] = 0 # Initialise p to 0 
    bounds = [(0, None)] * (2 * num_teams + 2)  # Adjust bounds

    print('Optimisation in Progress')
    # Run the optimization
    result = minimize(
        objective_function,
        initial_guess,
        args=(df,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 15}
    )
    # Output the optimized parameters
    optimised_parameters = result.x
    print(f'Optimisation Complete. Negative log likelihood = {result.fun}')
    
    # Save parameters as DataFrames
    df_attacking_scores = pd.DataFrame({
        'team': np.sort(df['home_team'].unique()),
        'attacking_score': optimised_parameters[0:num_teams]})
    df_defending_scores = pd.DataFrame({
        'team': np.sort(df['home_team'].unique()),
        'defending_score': optimised_parameters[num_teams:num_teams*2]})
    df_home_advantage = pd.DataFrame({
        'parameter': ['home_advantage', 'p'],
        'value': optimised_parameters[num_teams*2:num_teams*2+2]})

    # Save DataFrames as CSV file
    # Create a 'data' directory if it doesn't exist
    os.makedirs('data_'+league, exist_ok=True)
    
    # Save the DataFrames as CSV files in the 'data' folder
    attacking_scores_csv_path = os.path.join('data_'+league, 'attacking_scores.csv')
    df_attacking_scores.to_csv(attacking_scores_csv_path, index=False)
    print(f"Attacking Scores saved as {attacking_scores_csv_path}")
    
    defending_scores_csv_path = os.path.join('data_'+league, 'defending_scores.csv')
    df_defending_scores.to_csv(defending_scores_csv_path, index=False)
    print(f"Defending Scores saved as {defending_scores_csv_path}")
    
    home_advantage_csv_path = os.path.join('data_'+league, 'home_advantage.csv')
    df_home_advantage.to_csv(home_advantage_csv_path, index=False)
    print(f"Home Advantage saved as {home_advantage_csv_path}")
    return

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Estimate Attacking and Defending Score of teams")
    parser.add_argument('league', type=str, help='Specify which league')
    args = parser.parse_args()
    estimate_ad_score(args.league)
    

if __name__ == "__main__":
    main()

