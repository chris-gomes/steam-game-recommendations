import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def get_play_data():
    user_behavior = pd.read_csv("data\\steam-200k.csv", header=None).drop(columns=4)
    user_behavior.columns = ["user_id", "game_name", "behavior", "amount"]

    # get only play interactions
    user_plays = user_behavior[user_behavior["behavior"] == "play"].drop(columns="behavior")
    
    # make sure each user only has one play amount for each game
    user_plays = user_plays.groupby(["user_id", "game_name"]).sum().reset_index()

    

def main():
    pd.read_csv("https://www.kaggle.com/tamber/steam-video-games/download")