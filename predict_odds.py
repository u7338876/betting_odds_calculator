import numpy as np
import pandas as pd
import math
import argparse
import os

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

def score_probability(x, y, home_attack, home_defence, away_attack, away_defence, home_advantage, p):
    lambd = home_attack * away_defence * home_advantage
    mil = away_attack * home_defence
    t = calculate_t(x, y, lambd, mil, p)
    poisson_x = (lambd**x * np.exp(-lambd))/math.factorial(x)
    poisson_y = (mil**y * np.exp(-mil))/math.factorial(y)
    return t*poisson_x*poisson_y    

def get_probability_array(home_team, away_team, df_attack, df_defence, df_home_advantage):
    prob_array = np.zeros((9,9))
    total_probability = 0
    home_attack = df_attack.loc[home_team].attacking_score
    away_attack = df_attack.loc[away_team].attacking_score
    home_defence = df_defence.loc[home_team].defending_score
    away_defence = df_defence.loc[away_team].defending_score
    home_advantage = df_home_advantage.loc['home_advantage'].value
    p = df_home_advantage.loc['p'].value

    # Get probability of each scoreline up to 8 goals
    for i in range(9):
        for j in range(9):
            prob = score_probability(i, j, home_attack, home_defence, away_attack, away_defence, home_advantage, p)
            prob_array[i][j] = prob
            total_probability += prob
            
    prob_array /= total_probability # Normalise probabilities
    return prob_array

def match_odds(prob_array):
    home_win = 0
    draw = 0
    away_win = 0
    rows, cols = prob_array.shape
    for i in range(rows):
        for j in range(cols):
            if i > j:
                home_win += prob_array[i][j]
            elif j > i:
                away_win += prob_array[i][j]
            else: 
                draw += prob_array[i][j]
    home_win_odds = np.round(1/home_win,2)
    draw_odds = np.round(1/draw,2)
    away_win_odds = np.round(1/away_win,2)
    return [home_win_odds, draw_odds, away_win_odds]

def score_odds(prob_array):
    return pd.DataFrame(np.round(1/prob_array,2))

def over_under_odds(prob_array, goals):
    over = 0
    under = 0
    rows, cols = prob_array.shape
    for i in range(rows):
        for j in range(cols):
            if i+j > goals:
                over += prob_array[i][j]
            else:
                under += prob_array[i][j]
    over_odds = np.round(1/over,2)
    under_odds = np.round(1/under,2)
    return [over_odds, under_odds]

def both_to_score(prob_array):
    both_score = 0
    not_both_score = 0
    rows, cols = prob_array.shape
    for i in range(rows):
        for j in range(cols):
            if i == 0 or j == 0:
                not_both_score += prob_array[i][j]
            else: 
                both_score += prob_array[i][j]
    both_score_odds = np.round(1/both_score,2)
    not_both_score_odds = np.round(1/not_both_score,2)
    return [both_score_odds, not_both_score_odds]

def result_both_to_score(prob_array):
    home_win_both_score = 0
    home_win_not_both_score = 0
    draw_both_score = 0
    draw_not_both_score = 0
    away_win_both_score = 0
    away_win_not_both_score = 0
    rows, cols = prob_array.shape
    for i in range(rows):
        for j in range(cols):
            if i > j:
                if j == 0:
                    home_win_not_both_score += prob_array[i][j]
                else:
                    home_win_both_score += prob_array[i][j]
            elif j > i:
                if i == 0:
                    away_win_not_both_score += prob_array[i][j]
                else: 
                    away_win_both_score += prob_array[i][j]
            else:
                if j == 0:
                    draw_not_both_score += prob_array[i][j]
                else:
                    draw_both_score += prob_array[i][j]
    home_win_both_score_odds = np.round(1/home_win_both_score,2)
    home_win_not_both_score_odds = np.round(1/home_win_not_both_score,2)
    draw_both_score_odds = np.round(1/draw_both_score,2)
    draw_not_both_score_odds = np.round(1/draw_not_both_score,2)
    away_win_both_score_odds = np.round(1/away_win_both_score,2)
    away_win_not_both_score_odds = np.round(1/away_win_not_both_score,2)
    return [home_win_both_score_odds, home_win_not_both_score_odds, draw_both_score_odds, 
            draw_not_both_score_odds, away_win_both_score_odds, away_win_not_both_score_odds]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Betting Odds Calculator")
    
    # Positional arguments for team names
    parser.add_argument('league', type=str, help='Specify the league')
    parser.add_argument('home_team', type=str, help='Specify the home team')
    parser.add_argument('away_team', type=str, help='Specify the away team')
    
    # Add flags
    parser.add_argument('--match_odds', action='store_true', help='Calculate match odds')
    parser.add_argument('--score_odds', action='store_true', help='Calculate score odds')
    parser.add_argument('--over_under', type=float, help='Set over/under value')
    parser.add_argument('--both_to_score', action='store_true', help='Calculate odds of both teams scoring')
    parser.add_argument('--result_both_to_score', action='store_true', help='Calculate result and odds of both teams scoring')
    parser.add_argument('--all', action='store_true', help='Show all calculated odds')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Output the team names and the selected flags
    print(f"Home Team: {args.home_team}")
    print(f"Away Team: {args.away_team}")
    print("-------------------------------------------")
    
     # Read CSV files
    attacking_scores_csv_path = os.path.join('data_'+args.league, 'attacking_scores.csv')
    defending_scores_csv_path = os.path.join('data_'+args.league, 'defending_scores.csv')
    home_advantage_csv_path = os.path.join('data_'+args.league, 'home_advantage.csv')
    df_attack = pd.read_csv(attacking_scores_csv_path, index_col='team')
    df_defence = pd.read_csv(defending_scores_csv_path, index_col='team')
    df_home_advantage = pd.read_csv(home_advantage_csv_path, index_col='parameter')

    prob_array = get_probability_array(args.home_team, args.away_team, df_attack, df_defence, df_home_advantage)
    
    if args.match_odds:
        home_win, draw, away_win = match_odds(prob_array)
        print(f"{args.home_team} Winning Odds: {home_win}")
        print(f"Draw Odds: {draw}")
        print(f"{args.away_team} Winning Odds: {away_win}")
    
    if args.score_odds:
        print("Rows represent home team score and columns represent away team score")
        print(score_odds(prob_array))
    
    if args.over_under is not None:
        over, under = over_under_odds(prob_array, args.over_under)
        print(f"Over {args.over_under} Odds: {over}")
        print(f"Under {args.over_under} Odds: {under}")
    
    if args.both_to_score:
        both_score, not_both_score = both_to_score(prob_array)
        print(f"Both to Score Odds: {both_score}")
        print(f"Not Both to Score Odds: {not_both_score}")

    if args.result_both_to_score:
        home_win_both_score, home_win_not_both_score, draw_both_score, draw_not_both_score, away_win_both_score, away_win_not_both_score = result_both_to_score(prob_array)
        print(f"{args.home_team} to win and both to Score Odds: {home_win_both_score}")
        print(f"{args.home_team} to win and not both to Score Odds: {home_win_not_both_score}")
        print(f"Draw and both to Score Odds: {draw_both_score}")
        print(f"Draw and not both to Score Odds: {draw_not_both_score}")
        print(f"{args.away_team} to win and both to Score Odds: {away_win_both_score}")
        print(f"{args.away_team} to win and not both to Score Odds: {away_win_not_both_score}")
        
    if args.all:
        home_win, draw, away_win = match_odds(prob_array)
        print(f"{args.home_team} Winning Odds: {home_win}")
        print(f"Draw Odds: {draw}")
        print(f"{args.away_team} Winning Odds: {away_win}")
        print("-------------------------------------------")

        print("Rows represent home team score and columns represent away team score")
        print(score_odds(prob_array))
        print("-------------------------------------------")

        both_score, not_both_score = both_to_score(prob_array)
        print(f"Both to Score Odds: {both_score}")
        print(f"Not Both to Score Odds: {not_both_score}")
        print("-------------------------------------------")

        home_win_both_score, home_win_not_both_score, draw_both_score, draw_not_both_score, away_win_both_score, away_win_not_both_score = result_both_to_score(prob_array)
        print(f"{args.home_team} to win and both to Score Odds: {home_win_both_score}")
        print(f"{args.home_team} to win and not both to Score Odds: {home_win_not_both_score}")
        print(f"Draw and both to Score Odds: {draw_both_score}")
        print(f"Draw and not both to Score Odds: {draw_not_both_score}")
        print(f"{args.away_team} to win and both to Score Odds: {away_win_both_score}")
        print(f"{args.away_team} to win and not both to Score Odds: {away_win_not_both_score}")
        

if __name__ == "__main__":
    main()