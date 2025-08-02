from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load(r"C:\Users\Shriya\OneDrive\Documents\LAUNCHED\Machine Learning Capstone Project\3_flask_app\model.pkl")
template = pd.read_csv(r"C:\Users\Shriya\OneDrive\Documents\LAUNCHED\Machine Learning Capstone Project\2_model_training\input_template.csv")

template.columns = template.columns.str.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()

        form_data_clean = {k.strip(): v for k, v in form_data.items()}

        df = template.copy()

        for key, value in form_data_clean.items():
            if key in df.columns:
                df.at[0, key] = value

        df = df.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(df)[0]
        return render_template("index.html", prediction=round(prediction))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
