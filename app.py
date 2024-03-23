from flask import Flask, render_template, request

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

# Route for serving the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/prod')
def prod():
    return render_template('production.html')

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
    # Get water quality data from form
    water_quality_data = {}
    for parameter in thresholds.keys():
        water_quality_data[parameter] = float(request.form[parameter])
    
    # Generate recommendations based on the data
    recommendations = generate_recommendations(water_quality_data)
    
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
# from urllib import request
# from flask import Flask, render_template

# app = Flask(__name__)
# thresholds = {
#     'ph': {'min': 6.5, 'max': 8.5},
#     'hardness': {'min': 0, 'max': 200},
#     'solids': {'min': 0, 'max': 500},
#     'chloramine': {'min': 0, 'max': 4},
#     'sulphate': {'min': 0, 'max': 500},
#     'organic_carbon': {'min': 0, 'max': 50},
#     'trihalomethanes': {'min': 0, 'max': 100},
#     'turbidity': {'min': 0, 'max': 5}
# }
# # Route for serving the homepage
# @app.route('/')
# def home():
#     return render_template('index.html')

# def generate_recommendations(data):
#     recommendations = []
#     for parameter, value in data.items():
#         if parameter in thresholds:
#             if value < thresholds[parameter]['min']:
#                 recommendations.append(f"The {parameter} level is too low.")
#             elif value > thresholds[parameter]['max']:
#                 recommendations.append(f"The {parameter} level is too high.")
#     if not recommendations:
#         recommendations.append("Water quality is within acceptable range.")
#     return recommendations

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # Get water quality data from form
#     water_quality_data = {}
#     for parameter in thresholds.keys():
#         water_quality_data[parameter] = float(request.form[parameter])
    
#     # Generate recommendations based on the data
#     recommendations = generate_recommendations(water_quality_data)
    
#     return render_template('recommendations.html', recommendations=recommendations)
# if __name__ == '__main__':
#     app.run(debug=True)
