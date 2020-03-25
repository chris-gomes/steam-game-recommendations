import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def get_play_data():
    user_behavior = pd.read_csv(
        "data\\steam-200k.csv", header=None).drop(columns=4)
    user_behavior.columns = ["user_id", "game_name", "behavior", "amount"]

    # get only play interactions
    user_plays = user_behavior[user_behavior["behavior"] == "play"].drop(
        columns="behavior")

    # make sure each user only has one play amount for each game
    user_plays = user_plays.groupby(
        ["user_id", "game_name"]).sum().reset_index()

    return user_plays


def play_minimum(plays, threshold):
    games_played = plays.groupby("user_id").count().reset_index()
    games_played = games_played[(games_played['amount'] >= threshold)]
    new_plays = plays.merge(games_played["user_id"])
    return new_plays


def drop_too_popular(plays, threshold):
    users_played = plays.groupby("user_id").count().reset_index()
    users_played = users_played[(users_played['amount'] <= threshold)]
    new_plays = plays.merge(users_played['user_id'])
    return new_plays


def min_max_norm(plays):
    new_plays = plays.merge(plays.groupby('user_id').agg(
        {'amount': 'min'}).reset_index(), on='user_id')
    new_plays = new_plays.merge(new_plays.groupby('user_id').agg(
        {'amount_x': 'max'}).reset_index(), on='user_id')
    new_plays.columns = ['user_id', 'game_name', 'amount', 'min', 'max']
    new_plays['norm_amount'] = (new_plays['amount'] - new_plays['min'].apply(
        np.floor)) / (new_plays['max'] - new_plays['min'].apply(np.floor))
    return new_plays.drop(columns=['min', 'max'])


def main():
    plays = get_play_data()

    # only keep play times of at least 1 hour
    plays = plays[plays['amount'] >= 1]

    # only keep users that played at least 5 games
    plays = play_minimum(plays, 5)

    # remove games that more than 1000 people have played
    plays = drop_too_popular(plays, 1000)

    # normalize the play amounts by the users min and max play times per game
    plays = min_max_norm(plays)

    # split into train and test
    train, test = train_test_split(plays, test_size=0.25)

    # save to csv
    train.to_csv("data\\train-plays.csv", index=False)
    test.to_csv("data\\test-plays.csv", index=False)


main()
