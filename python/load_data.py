import pandas as pd
from surprise import Dataset
from surprise import Reader

ratings_dict = {
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))

# Load df
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# Load movie daata
movie = Dataset.load_from_file('../data/Movie Data/ml-100k/u.data', reader)