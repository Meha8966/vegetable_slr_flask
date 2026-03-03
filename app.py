from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_value = float(request.form["feature"])
    prediction = model.predict([[input_value]])
    return render_template("home.html", result=prediction[0][0])

# Allow GET and POST for login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        # For now, just redirect to home after "login"
        return redirect(url_for("home"))
    return render_template("login.html")

# Allow GET and POST for register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        # For now, just redirect to login after "registration"
        return redirect(url_for("login"))
    return render_template("register.html")

if __name__ == "__main__":
    app.run(debug=True)