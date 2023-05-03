from flask import Flask, render_template, request, jsonify
import test_h5

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/display", methods=["POST", "GET"])
def display():
    return render_template("output.html")

@app.route("/output", methods=["POST", "GET"])
def output(): 
    inp = request.form.get("inp")
    prediction = test_h5.foo(inp)
    return render_template('finalout.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
