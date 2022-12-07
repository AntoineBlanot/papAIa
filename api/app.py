from flask import Flask, render_template, request, flash, url_for, redirect

import pandas as pd
from recom import get_vectors, topk, missing_names

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'

# To launch the hello world api, do in cmd: 
## export FLASK_APP=api/app
## export FLASK_ENV=development
## flask run
## CTRL+C to stop 
# Go to http://127.0.0.1:5000/

DEFAULT_INPUTS = {
    "k": 5,
    "ingredient[]": []
}
SHOPPING_CART = []
RESULT = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/pantry/')
def pantry():
    return render_template('pantry.html')

@app.route('/shopping_lists/')
def shopping_lists():
    return render_template('shopping_lists.html')

@app.route('/ingredients/', methods=('GET', 'POST'))
def ingredients():
    if request.method == 'POST':

        # Request the user input
        k = int(request.form['k']) if request.form['k'] != '' else DEFAULT_INPUTS["k"]
        ing = request.form['ingredient[]'].replace(' ', '').split(',') if request.form['ingredient[]'] != '' else DEFAULT_INPUTS["ingredient[]"]
        inputs = dict(k=k, ingredients=ing)

        print(inputs)

        if len(inputs["ingredients"]) == 0:
            flash('Please enter at least one ingredient !', category='warning')
        else:
            return recomendation(inputs=inputs)

    return render_template('ingredients.html')

@app.route('/recomendation/', methods=('GET', 'POST'))
def recomendation(inputs: dict):
    result = recommend(inputs=inputs)
    # if request.method == 'POST':
    #     i = int(request.form['i']) if request.form['i'] != '' else None
    #     SHOPPING_CART.extend(result.iloc[i].missing_name.tolist())
    
    return render_template("recomendation.html", tables=[result.to_html(classes='data')], titles=result.columns.values, cart=SHOPPING_CART)
            
def recommend(inputs: dict):
    user_data = pd.DataFrame(dict(title=[""], NER=[inputs["ingredients"]]))
    vector = get_vectors(data=user_data)
    result = topk(vector, k=inputs["k"])
    result["missing_name"] = missing_names(missing_list=result["missing_id"].values)
    return result