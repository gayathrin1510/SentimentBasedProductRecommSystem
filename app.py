## Date: 12th March 2022
## Author: GAYATHRI N
# SENTIMENT BASED PRODUCT RECOMMENDATION SYSTEM

import model
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        user = str(request.form['username'])
        productRecommendations = model.get_sentimentBasedProductRecommendations(user)
        return render_template('index.html', products =productRecommendations, users=model.UserList)
    else:
        return render_template('index.html', products =[], users=model.UserList)

# Run application
if __name__ == "__main__":
    app.run(debug=True)
