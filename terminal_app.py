import pandas as pd

from exploration.process import get_vectors
from exploration.metrics import topk

DATA_FOLDER = "data/"
SUBSET = 0.01
STOP = "-1"

data = pd.read_csv("./data/vector-0.01.csv")
mapping = pd.read_csv(f"{DATA_FOLDER}mapping-{SUBSET}.csv", sep=",")

AVAILABLE_INGREDIENTS = []
SHOPPING_CART = []

def ask():
    end = False
    new_ingredients = []
    print("Input your current ingredients ({} to stop)".format(STOP))

    while not end:
        i = input()
        if i == "-1":
            end = True
        else:
            new_ingredients.append(i)
    
    # AVAILABLE_INGREDIENTS.extend(new_ingredients)
    # AVAILABLE_INGREDIENTS = list(set(AVAILABLE_INGREDIENTS))
    print("You currently have {} ingredients:\n{}".format(len(new_ingredients), new_ingredients))
    
    return new_ingredients

def missing_names(missing_list: list, mapping: pd.DataFrame):
    names = [
        mapping.iloc[missing].ingredient.values
        for missing in missing_list
    ]
    return names

def main():
    while True:
        ings = ask()
        vector = get_vectors(data=pd.DataFrame(dict(title=[""], NER=[ings])), mapping=mapping)
        result = topk(vector, data=data, k=20)    
        result["missing_name"] = missing_names(missing_list=result["missing_id"].values, mapping=mapping)
        print(result.head(20))

        meal_idx = int(input("Please select the meal by its index: "))
        SHOPPING_CART.extend(result.iloc[meal_idx].missing_name.tolist())
        print("Your new shopping cart is: {}".format(SHOPPING_CART))

if __name__ == "__main__":
    main()