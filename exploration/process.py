from typing import List

from ast import literal_eval
import numpy as np
import pandas as pd


def read_data(file: str) -> pd.DataFrame:
    """Read the data file"""
    data = pd.read_csv(file, converters=dict(NER=literal_eval), index_col=0)
    print(data.info())
    return data


def subset(data: pd.DataFrame, proportion: float) -> pd.DataFrame:
    """Return a subset of the data"""
    size = int(len(data) * proportion)
    return data.iloc[ : size]


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data"""
    # remove the data where ingredients are an empty list
    data = data.loc[data["NER"].apply(len) != 0].reset_index(drop=True) 
    return data


def get_ingredients(data: pd.DataFrame) -> List:
    """Retrieve the list of unique ingredients"""
    ingredients = sum(data["NER"].values.tolist(), [])    # unnest the list of list to list
    return list(set(ingredients))                       # retrieve unique elements 


def get_mapping(ingredients: List) -> pd.DataFrame:
    """Return the mapping dataframe where each ingredient is mapped to an index"""
    # map each ingredient to a single index (number)
    mapping = pd.DataFrame(dict(index=range(len(ingredients)), ingredient=ingredients))
    return mapping


def get_vectors(data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with the data vectors based on the mapping"""
    dummies = pd.get_dummies(mapping.ingredient)

    def convert_to_vectors(row):
        ingredient_list = row["NER"]
        vector_list = [np.expand_dims(dummies[i].values, axis=0) for i in ingredient_list]
        return np.concatenate(vector_list, axis=0).sum(axis=0, keepdims=True)
    
    vectors = data.apply(convert_to_vectors, axis=1)
    vectors = np.concatenate(vectors.values.tolist(), axis=0)

    return pd.DataFrame(data=vectors, columns=mapping.ingredient.values, index=data["title"]).reset_index()


if __name__ == "__main__":

    DATA_FOLDER = "data/"
    SUBSET = 0.01

    data = read_data(file=f"{DATA_FOLDER}meals.csv")
    data = subset(data=data, proportion=SUBSET)
    data = clean(data=data)
    data.to_csv(f"{DATA_FOLDER}meals-{SUBSET}.csv", sep=",", index=False)
    print("Data cleaned")

    ingredients = get_ingredients(data=data)
    print("Got ingredients")
    mapping = get_mapping(ingredients=ingredients)
    print("Got mapping")
    mapping.to_csv(f"{DATA_FOLDER}mapping-{SUBSET}.csv", sep=",", index=False)

    vectors = get_vectors(data=data, mapping=mapping)
    print("Got vectors")
    vectors.to_csv(f"{DATA_FOLDER}vector-{SUBSET}.csv", sep=",", index=False)
