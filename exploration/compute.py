import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def topk(vector, data, k=5):
    sim = cosine_similarity(vector, data.iloc[:, 1:].values)
    topk = (-sim).argsort()[:, :k]

    scores = sim[:, topk[0]]
    meals = data.iloc[topk[0]].title
    ingredients = data.iloc[topk[0], 1:]

    diff = ingredients.values - vector
    missing_list = []
    for i in range(len(diff)):
        missing = np.where(diff[i] == 1)[0]
        missing_list.append(missing)

    return meals, scores, missing_list