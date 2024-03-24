from flask import Flask, render_template, request
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
thresholds = {
    'ph': {'min': 6.5, 'max': 8.5},
    'hardness': {'min': 0, 'max': 200},
    'solids': {'min': 0, 'max': 500},
    'chloramine': {'min': 0, 'max': 4},
    'sulphate': {'min': 0, 'max': 500},
    'organic_carbon': {'min': 0, 'max': 50},
    'trihalomethanes': {'min': 0, 'max': 100},
    'turbidity': {'min': 0, 'max': 5}
}
dataset = pd.read_csv('plant_health_dataset.csv')

X = dataset.drop('HealthStatus', axis=1)
y = dataset['HealthStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        leaf_size = request.form['leaf_size']
        stem_thickness = request.form['stem_thickness']
        blossom_size = request.form['blossom_size']
       
        input_data = [[ leaf_size, stem_thickness, blossom_size]]
        input_df = pd.DataFrame(input_data, columns=X.columns)

        prediction = model.predict(input_df)

        return render_template('predict.html', prediction=prediction[0])
# Route for serving the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

def generate_recommendations(data):
    recommendations = []
    for parameter, value in data.items():
        if parameter in thresholds:
            if value < thresholds[parameter]['min']:
                recommendations.append(f"The {parameter} level is too low.")
            elif value > thresholds[parameter]['max']:
                recommendations.append(f"The {parameter} level is too high.")
    if not recommendations:
        recommendations.append("Water quality is within acceptable range.")
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    water_quality_data = {}
    for parameter in thresholds.keys():
        water_quality_data[parameter] = float(request.form[parameter])
    
    recommendations = generate_recommendations(water_quality_data)
    
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
