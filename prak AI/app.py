from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    name = None
    message = None

    if request.method == "POST":
        name = request.form["name"]
        message = request.form["message"]

    return render_template("index.html", name=name, message=message)

if __name__ == "__main__":
    app.run(debug=True)
