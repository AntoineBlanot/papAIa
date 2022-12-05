import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

def topk(vector: pd.DataFrame, data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    vector = vector.iloc[:, 1:].values
    sim = cosine_similarity(vector, data.iloc[:, 1:].values)
    topk = (-sim).argsort()[:, :k]

    scores = sim[:, topk[0]][0]
    meals = data.iloc[topk[0]]["title"].values
    ingredients = data.iloc[topk[0], 1:]

    diff = ingredients.values - vector
    missing_list = []
    for i in range(len(diff)):
        missing = np.where(diff[i] == 1)[0]
        missing_list.append(missing)

    res = pd.DataFrame(dict(name=meals, score=scores, missing_id=missing_list))

    return res