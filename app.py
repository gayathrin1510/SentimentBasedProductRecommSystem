import model
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        user = str(request.form['usersList'])
        productRecommendations = model.get_sentimentBasedProductRecommendations(user)
        return render_template('index.html', products =productRecommendations, users=model.userList)
    else:
        return render_template('index.html', products =[], users=model.userList)

# Run application
if __name__ == "__main__":
    app.run(debug=True)
