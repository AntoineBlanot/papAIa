import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = "./data/"
SUBSET = 0.01
data = pd.read_csv(f"{DATA_FOLDER}vector-0.01.csv")
mapping = pd.read_csv(f"{DATA_FOLDER}mapping-{SUBSET}.csv", sep=",")


def get_vectors(data: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with the data vectors based on the mapping"""
    dummies = pd.get_dummies(mapping.ingredient)

    def convert_to_vectors(row):
        ingredient_list = row["NER"]
        vector_list = [np.expand_dims(dummies[i].values, axis=0) for i in ingredient_list]
        return np.concatenate(vector_list, axis=0).sum(axis=0, keepdims=True)
    
    vectors = data.apply(convert_to_vectors, axis=1)
    vectors = np.concatenate(vectors.values.tolist(), axis=0)

    return pd.DataFrame(data=vectors, columns=mapping.ingredient.values, index=data["title"]).reset_index()

def topk(vector: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    vector = vector.iloc[:, 1:].values
    sim = cosine_similarity(vector, data.iloc[:, 1:].values)
    topk = (-sim).argsort()[:, :k]

    sim_scores = sim[:, topk[0]][0]
    meals = data.iloc[topk[0]]["title"].values
    ingredients = data.iloc[topk[0], 1:]

    diff = ingredients.values - vector
    missing_list = []
    for i in range(len(diff)):
        missing = np.where(diff[i] == 1)[0]
        missing_list.append(missing)

    scores = sim_scores - np.array([len(m) * 0.1 for m in missing_list])
    res = pd.DataFrame(dict(name=meals, sim_score=sim_scores, score=scores, missing_id=missing_list))
    res = res.sort_values(by="score", ascending=False).reset_index(drop=True)

    return res

def missing_names(missing_list: list):
    names = [
        mapping.iloc[missing].ingredient.values
        for missing in missing_list
    ]
    return names
