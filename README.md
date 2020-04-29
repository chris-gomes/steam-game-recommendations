# Steam Game Recommendations

The goal of this project is to use collaborative filtering methods to build recommendation systems for recommending Steam games for users based on games they previously played. A write-up for the initial findings is available [here](https://ccgomes.com/files/steam_game_recommendations.pdf).

## Getting Started
1. Clone this repo
1. pip install the libraries in `requirements.txt`
1. Download csv file from https://www.kaggle.com/tamber/steam-video-games/data to a directory called "data" in the cloned repo
1. Run `preprocessing.py` to get the cleaned user information
1. Use the existing notebooks or new ones to explore the cleaned data files in the "data" directory

## Models
- Alternate Least Squares (http://yifanhu.net/PUB/cf.pdf)
- Bayesian Personalized Ranking (https://arxiv.org/pdf/1205.2618.pdf)
- Logistic Matrix Factorization (https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
- Neural Collaborative Filtering (https://arxiv.org/pdf/1708.05031.pdf)

## Datasets
- User Game Purchases: https://www.kaggle.com/tamber/steam-video-games/data

## Libraries
- Implicit (https://implicit.readthedocs.io/en/latest/index.html)
- TensorFlow (https://www.tensorflow.org/)

## Authors

* **Christopher Gomes** - [chris-gomes](https://github.com/chris-gomes)
* **Steven Horn** - [steven-horn](https://github.com/steven-horn)
