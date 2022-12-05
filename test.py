import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# from exploration.process import get_vectors
from exploration.compute import topk

DATA_FOLDER = "data/"
SUBSET = 0.01

data = pd.read_csv("./data/vector-0.01.csv")
mapping = pd.read_csv(f"{DATA_FOLDER}mapping-{SUBSET}.csv", sep=",")
dummies = pd.get_dummies(mapping.ingredient)

end = False
l = []
print("Add ingredients (-1 to stop)")
while not end:
    i = input()
    if i == "-1":
        end = True
    else:
        l.append(i)

print(l)
# d = pd.DataFrame(dict(NER=[l], title=["input"]))
# print(d.head())
# # convert to vector
v = [np.expand_dims(dummies[i].values, axis=0) for i in l]
v = np.concatenate(v, axis=0).sum(axis=0, keepdims=True)
# v = get_vectors(data=d, mapping=mapping)
# distance 
print(v)
print(v.shape)

print(mapping.loc[mapping.ingredient == "tomato"])
print(mapping.loc[mapping.ingredient == "chicken"])
print(np.where(v == 1))

meals, scores, missing_list = topk(v, data=data, k=10)

print(meals)
print(scores)
print(missing_list)