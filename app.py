from flask import Flask, render_template, request
from model.predictor import predict_stress

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    level = ""
    user_input = ""
    if request.method == "POST":
        user_input = request.form["text"]
        prediction, level = predict_stress(user_input)
    return render_template("index.html", prediction=prediction, level=level, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
