# betting_odds_calculator

## Introduction

Welcome to my project that aims to identify good football betting odds, based on past fixtures. Currently, this project is only limited to Europe's top five football leagues: (1) English Premier League, (2) La Liga, (3) Serie A, (4) Bundesliga, and (5) Ligue 1. The model references the work of Dixon and Coles (1997). This project is meant for my own personal interest, and if you wish to use it for betting, do so at your own risk. Data used was scraped from [https://understat.com/](https://understat.com/). 

## Training

The code utilises a Poisson distribution to model the likelihood of different scorelines in a football match between two teams. It optimises parameters by minimising the negative log likelihood, ensuring the best fit to observed data, in order to accurately predict the probability of each possible scoreline. The training phase uses the scorelines of the previous season (2023/24) to predict how good a team's attack and defence is, as well as the effect of home advantage. Run the code below to save these attacking and defending scores as CSV files. 

For English Premier League teams: 
```
python estimate_ad_score.py "EPL"
```

For La Liga teams: 
```
python estimate_ad_score.py "LaLiga"
```

For Serie A teams: 
```
python estimate_ad_score.py "SerieA"
```

For Bundesliga teams: 
```
python estimate_ad_score.py "Bundesliga"
```

For Ligue 1 teams: 
```
python estimate_ad_score.py "Ligue1"
```

## Predicting

To predict odds, run the [predict_odds.py](https://github.com/u7338876/betting_odds_calculator/blob/main/predict_odds.py) file with the following arguments: 
- league: "EPL"/"LaLiga"/"SerieA"/"Bundesliga"/"Ligue1"
- home_team:
- away_team:
- flag: --score_odds/--over_under/--both_to_score/--result_both_to_score/--all

For example, 
```
python predict_odds.py "EPL" "Manchester City" "Brentford" --all
```

## Limitations

As the training phase uses data from the previous season of the top leagues, odds cannot be predicted for newly promoted teams such as Leicester City, Southampton, and Ipswich (for EPL). 
