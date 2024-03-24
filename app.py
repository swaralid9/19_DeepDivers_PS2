from flask import Flask, render_template, request
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import random
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

dataset_path = 'PlantDisease'  


def get_train_generator():
    train_path = os.path.join(dataset_path, 'train')
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(100, 100),
        batch_size=32,
        class_mode='binary'  # Assuming two classes: healthy and unhealthy
    )
    return train_generator

def classify_random_image():
    train_generator = get_train_generator()

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification: healthy or unhealthy
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_generator, epochs=15)

    random_class = random.choice(['healthy', 'unhealthy'])
    class_path = os.path.join(dataset_path, 'train', random_class)
    if os.listdir(class_path):
        random_image_filename = random.choice(os.listdir(class_path))
        random_image_path = os.path.join(class_path, random_image_filename)
        random_image = load_img(random_image_path, target_size=(100, 100))

        random_image_array = img_to_array(random_image)
        random_image_array = random_image_array.reshape(1, 100, 100, 3) / 255.0

        prediction = model.predict(random_image_array)
        predicted_class = 'healthy' if prediction < 0.5 else 'unhealthy'

        predictions_data = {
            'image_path': random_image_path,
            'predicted_class': predicted_class,
        }
    else:
        predictions_data = {'image_path': '', 'predicted_class': 'Unknown'}

    return predictions_data

@app.route('/')
def index():
    predictions_data = {'image_path': '', 'predicted_class': 'Unknown'}
    return render_template('index.html', predictions_data=predictions_data)

@app.route('/results')
def results():
    predictions_data = classify_random_image()
    return render_template('results.html', predictions_data=predictions_data)


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
       # model and train_generator
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')  # 7 classes including 'Endangered_Animals'
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_generator = get_train_generator()

    app.run(debug=True)
