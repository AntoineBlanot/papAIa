import pandas as pd

from exploration.process import get_vectors
from exploration.metrics import topk

DATA_FOLDER = "data/"
SUBSET = 0.01
STOP = "-1"

data = pd.read_csv("./data/vector-0.01.csv")
mapping = pd.read_csv(f"{DATA_FOLDER}mapping-{SUBSET}.csv", sep=",")


def ask():
    end = False
    current_ingredients = []
    print("Input your current ingredients ({} to stop)".format(STOP))

    while not end:
        i = input()
        if i == "-1":
            end = True
        else:
            current_ingredients.append(i)
    print("You currently have {} ingredients:\n{}".format(len(current_ingredients), current_ingredients))
    
    return current_ingredients

def missing_names(missing_list: list, mapping: pd.DataFrame):
    names = [
        mapping.iloc[missing].ingredient.values
        for missing in missing_list
    ]
    return names

def main():
    ings = ask()
    vector = get_vectors(data=pd.DataFrame(dict(title=[""], NER=[ings])), mapping=mapping)
    result = topk(vector, data=data, k=10)    
    result["missing_name"] = missing_names(missing_list=result["missing_id"].values, mapping=mapping)
    print(result.head())


if __name__ == "__main__":
    main()