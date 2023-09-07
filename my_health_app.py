from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset (replace with your actual dataset)
data = pd.read_csv('patient_data.csv')

# Features and target variable
X = data.drop('Health_Status', axis=1)
y = data['Health_Status']

# Train a Random Forest classifier (replace with the appropriate model)
model = RandomForestClassifier()
model.fit(X, y)


@app.route('/', methods=['GET', 'POST'])
def index():
    health_status = None
    if request.method == 'POST':
        patient_details = [
            int(request.form['age']),
            int(request.form['gender']),
            int(request.form['BP']),
            int(request.form['Heartbeat']),
            int(request.form['Sugar_Level']),
            int(request.form['Previous_Disease'])
        ]

        input_data = [patient_details]
        health_status = model.predict(input_data)[0]

    return render_template('index1.html', health_status=health_status)


if __name__ == '__main__':
    app.run(debug=True)
