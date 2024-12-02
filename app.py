import pickle
import numpy as np
from flask import Flask, jsonify, render_template, flash, redirect, url_for, request, session
from flask_cors import CORS
import requests
import pymysql.cursors
from werkzeug.security import generate_password_hash, check_password_hash
import math
from collections import defaultdict, Counter
from datetime import datetime, date
import os

app = Flask(__name__)
CORS(app)
app.secret_key = '69ff3699752ff683d2bcca765de29aca'

db = pymysql.connect(
    host="autorack.proxy.rlwy.net",  # Your Railway MySQL host
    user="root",                     # Your MySQL username
    password="BJxWrCOjaFlXAuTdthieITxmzlgGuoND",  # Your MySQL password
    database="railway",              # Your MySQL database name
    port=24341,
    cursorclass=pymysql.cursors.DictCursor # Your Railway MySQL port

)


WEATHER_BASE_URL = 'http://api.openweathermap.org/data/2.5'
API_KEY = '0441c36366b282b3fdbafb63653ca358'
CITY = 'Bacoor'
COUNTRY = 'PH'

with open('fog_prediction_model.pkl', 'rb') as model_file:
    loaded_data = pickle.load(model_file)
    if isinstance(loaded_data, dict):
        fog_model = loaded_data['model']
        scaler = loaded_data['scaler']
        optimal_threshold = loaded_data['threshold']
    else:
        raise ValueError("Loaded data is not in the expected format. Ensure the model was saved as a dictionary.")

def calculate_dew_point(T, RH):
    """
    Calculate dew point using Magnus formula
    T: Temperature in Celsius
    RH: Relative Humidity in percentage (0-100)
    Returns: Dew point in Celsius
    """
    # Constants for Magnus formula
    a = 17.27
    b = 237.7

    # Calculate gamma using relative humidity
    gamma = (a * T / (b + T)) + math.log(RH / 100.0)
    # Calculate dew point
    dew_point = (b * gamma) / (a - gamma)

    return round(dew_point, 2)

@app.route('/')
def home():
    return render_template('landingpage.html')

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/index_data')
def index_data():
    # Construct the URL to fetch the weather data
    url = f"{WEATHER_BASE_URL}/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"

    try:
        # Send the GET request to the OpenWeather API
        response = requests.get(url)

        # Check if the response status is OK (200)
        if response.status_code == 200:
            data = response.json()

            # Extract the relevant data from the response
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            rainfall_mm = data.get('rain', {}).get('1h', 0) or data.get('rain', {}).get('3h', 0)
            description = data['weather'][0].get('description', "N/A")  # Default to "N/A" if not available

            # If dew_point is not provided, calculate it (optional)
            dew_point = data['main'].get('dew_point')
            if dew_point is None:
                dew_point = calculate_dew_point(temperature, humidity)
            area_m2 = 50    # Bacoor area in square meters
            liters_of_rain = rainfall_mm * area_m2 * 0.8 # Convert rainfall from mm to liters

            # Extract other necessary weather features
            windspeed = data['wind']['speed']
            visibility = data['visibility'] / 1000  # Convert visibility to km
            cloudcover = data['clouds']['all']
            feelslike = data['main']['feels_like']

            # Prepare the feature vector for model input
            features = np.array([[temperature, dew_point, humidity, windspeed, visibility, cloudcover, feelslike]])

            # Scale the features using the loaded scaler
            scaled_features = scaler.transform(features)

            # Predict fog likelihood using the trained machine learning model
            prediction_prob = fog_model.predict_proba(scaled_features)[0][1]  # Probability of 'fog' class

            # Calculate fog likelihood as a percentage
            fog_likelihood_percentage = round(prediction_prob * 100, 2)

            # Return the data as a JSON response with fog likelihood prediction
            return jsonify({
                "temperature": f"{temperature}°C",
                "humidity": humidity,
                "rainfall_mm": rainfall_mm,
                "description": description,
                "dew_point": dew_point,
                "harvest": round(liters_of_rain, 2),  # Replace with actual logic if needed
                "fogLikelihood": fog_likelihood_percentage,  # Fog likelihood based on ML model prediction

            })
        else:
            # Return an error message if the API request fails
            return jsonify({'error': 'Could not fetch weather data'}), response.status_code
    except Exception as e:
        # Return an error message if something goes wrong with the request
        return jsonify({'error': str(e)}), 500



@app.route('/forecast_data', methods=['GET'])
def forecast_data():
    forecast_url = f'{WEATHER_BASE_URL}/forecast?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric'

    try:
        # Fetch forecast data from OpenWeather
        response = requests.get(forecast_url)
        response.raise_for_status()
        data = response.json()

        if data['cod'] == '200':
            # Group data by date
            daily_data = defaultdict(list)
            for item in data['list']:
                forecast_date = item['dt_txt'].split(' ')[0]  # Extract date
                temp = item['main']['temp']
                description = item['weather'][0]['description']
                rainfall_mm = item.get('rain', {}).get('3h', 0)  # Get 3-hour rainfall (in mm)

                # Store data by date
                daily_data[forecast_date].append((temp, description, rainfall_mm))

            # Dynamic programming table for rainfall sums and counts
            rainfall_sum = {}
            rainfall_count = {}

            # Process daily data and include today's forecast
            processed_forecast = []
            today = date.today()

            for forecast_date, values in sorted(daily_data.items()):
                # Calculate averages using stored sums and counts
                if forecast_date not in rainfall_sum:
                    rainfall_sum[forecast_date] = sum(rainfall for _, _, rainfall in values if rainfall > 0)
                    rainfall_count[forecast_date] = sum(1 for _, _, rainfall in values if rainfall > 0)

                avg_temp = round(sum(temp for temp, _, _ in values) / len(values))  # Average temperature
                avg_rainfall = (
                    round(rainfall_sum[forecast_date] / rainfall_count[forecast_date], 2)
                    if rainfall_count[forecast_date] > 0 else 0
                )

                # Calculate harvest (liters of rain)
                area_m2 = 50  # Assuming a fixed area of 50 m²
                liters_of_rain = avg_rainfall * area_m2 * 0.8  # 80% efficiency for rainwater harvesting
                harvest = round(liters_of_rain, 2)

                # Select the most common description
                description_counts = Counter(desc for _, desc, _ in values)
                descriptions_sorted = description_counts.most_common()

                # Find the most common description
                if descriptions_sorted:
                    most_common = descriptions_sorted[0][0].capitalize()
                else:
                    most_common = "No significant weather data"

                # Prioritize rain for the second most description
                second_common = None
                for desc, _ in descriptions_sorted[1:]:
                    if "rain" in desc.lower():
                        second_common = desc.capitalize()
                        break

                # If no rain is found, take the next most common description
                if not second_common and len(descriptions_sorted) > 1:
                    second_common = descriptions_sorted[1][0].capitalize()

                # Prepare the description string
                if second_common:
                    description_str = f"{most_common} | {second_common}"
                else:
                    description_str = most_common

                # Convert date to day and formatted date
                day_name = datetime.strptime(forecast_date, "%Y-%m-%d").strftime("%a")
                formatted_date = datetime.strptime(forecast_date, "%Y-%m-%d").strftime("%b %d")

                # Append the processed data
                processed_forecast.append({
                    "day": day_name,
                    "date": formatted_date,
                    "condition": description_str,  # Use the formatted description string
                    "temp": f"{avg_temp}°",
                    "avg_rainfall_mm": avg_rainfall,
                    "harvest_liters": harvest
                })

                # Stop processing after including today's and the next 4 unique days
                if len(processed_forecast) == 5:
                    break

            return jsonify(processed_forecast)

        else:
            return jsonify({"error": data.get('message', 'Unable to fetch forecast data')}), 400

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"error": f"Invalid response format: Missing key {str(e)}"}), 500

@app.route('/current_date', methods=['GET'])
def current_date():
    # Get the current date and day
    now = datetime.now()

    # Format the date
    current_day = now.strftime("%A")  # e.g., Monday
    current_date = now.strftime("%d %b %Y")  # e.g., 25 Nov 2024

    # Return day and date as JSON
    return jsonify({
        'day': current_day,
        'full_date': current_date
    })

@app.route('/wind')
def get_wind_info():
    url = f"{WEATHER_BASE_URL}/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            wind_speed = data['wind']['speed']
            wind_dir = data['wind'].get('deg', None)

            return jsonify({
                "wind_speed": wind_speed,
                "wind_dir": wind_dir
            })
        else:
            return jsonify({'error': 'Could not fetch wind data'}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_fog', methods=['GET'])
def predict_fog():
    def divide_and_conquer_weather(data):
        """
        Recursive function to process weather data using divide and conquer.
        Each part computes specific features and merges them.
        """
        if len(data) == 1:
            key, value = next(iter(data.items()))
            # Handle individual weather data features
            if key == 'temp':
                return {"temperature": value}
            elif key == 'humidity':
                return {"humidity": value}
            elif key == 'wind_speed':
                return {"windspeed": value}
            elif key == 'visibility':
                return {"visibility": value / 1000}  # Convert visibility to km
            elif key == 'cloudcover':
                return {"cloudcover": value}
            elif key == 'feelslike':
                return {"feelslike": value}
            elif key == 'rain_mm':
                return {"rain_mm": value}  # Add rain_mm if available
            else:
                return {}

        # Divide into two halves
        mid = len(data) // 2
        left = dict(list(data.items())[:mid])
        right = dict(list(data.items())[mid:])

        # Conquer both halves
        left_result = divide_and_conquer_weather(left)
        right_result = divide_and_conquer_weather(right)

        # Combine results
        return {**left_result, **right_result}

    # Fetch the current weather data from OpenWeather
    complete_url = f"{WEATHER_BASE_URL}/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"
    response = requests.get(complete_url)

    if response.status_code != 200:
        return jsonify({"error": "Unable to fetch weather data"}), 400

    data = response.json()

    # Raw weather data
    raw_features = {
        "temp": data['main']['temp'],
        "humidity": data['main']['humidity'],
        "wind_speed": data['wind']['speed'],
        "visibility": data['visibility'],
        "cloudcover": data['clouds']['all'],
        "feelslike": data['main']['feels_like']
    }

    # Check for rain data (1h > 3h > 0)
    rain_mm = 0
    if 'rain' in data:
        if '1h' in data['rain']:
            rain_mm = data['rain']['1h']
        elif '3h' in data['rain']:
            rain_mm = data['rain']['3h']

    # Add rain_mm to the raw features
    raw_features['rain_mm'] = rain_mm

    # Process features using divide and conquer
    processed_features = divide_and_conquer_weather(raw_features)

    # Calculate dew point
    processed_features['dew_point'] = calculate_dew_point(
        processed_features['temperature'], processed_features['humidity']
    )

    # Scale features (excluding rain_mm)
    feature_array = np.array([
        processed_features['temperature'],
        processed_features['dew_point'],
        processed_features['humidity'],
        processed_features['windspeed'],
        processed_features['visibility'],
        processed_features['cloudcover'],
        processed_features['feelslike']
    ]).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)

    # Predict fog likelihood
    prediction_prob = fog_model.predict_proba(scaled_features)[0][1]
    fog_prediction = 1 if prediction_prob >= optimal_threshold else 0
    fog_likelihood_percentage = round(prediction_prob * 100, 2)

    # Add the rain_mm to the final output (without affecting scaling)
    processed_features['rain_mm'] = rain_mm

    # Determine alerts
    rain_alert =bool (rain_mm > 0 ) # True if rain_mm > 0
    fog_alert =bool (fog_likelihood_percentage >= 30 ) # True if fog likelihood >= 30%

    # If no alerts are active, return the message saying no alerts
    if not rain_alert and not fog_alert:
        return jsonify({
            'message': "There is no alert at the moment.",
            'rain_alert': rain_alert,
            'fog_alert': fog_alert
        })

    # Return the response with all the necessary data
    return jsonify({
        **processed_features,
        'fog_prediction': fog_prediction,
        'fog_probability': prediction_prob,
        'fog_likelihood_percentage': fog_likelihood_percentage,
        'rain_alert': rain_alert,
        'fog_alert': fog_alert
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'register' in request.form:
            # Registration logic
            email = request.form['email']
            username = request.form['username']
            password = request.form['password']
            hashed_password = generate_password_hash(password)

            cursor = db.cursor()

            try:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    flash('Username already exists!')
                    return redirect(url_for('login'))

                sql = "INSERT INTO users (email, username, password) VALUES (%s, %s, %s)"
                values = (email, username, hashed_password)
                cursor.execute(sql, values)
                db.commit()

                flash('Registration successful! Please login.')
                return redirect(url_for('login'))

            except pymysql.Error as err:
                flash(f"An error occurred: {err}")
                return redirect(url_for('login'))

            finally:
                cursor.close()

        elif 'login' in request.form:
            # Login logic
            username = request.form['login_username']
            password = request.form['login_password']

            cursor = db.cursor()  # Just call cursor() - no need for dictionary=True
            try:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

                if user and check_password_hash(user['password'], password):
                    session['username'] = username  # Set session or login management
                    flash('Login successful!')
                    return redirect(url_for('index'))  # Replace 'index' with your home route

                else:
                    flash('Invalid username or password!')
                    return redirect(url_for('login'))

            except pymysql.Error as err:
                flash(f"An error occurred: {err}")
                return redirect(url_for('login'))

            finally:
                cursor.close()

    return render_template('login.html')



@app.route('/logout')
def logout():
    # Clear session data (if any)
    session.clear()
    # Redirect to landing page
    return redirect(url_for('home'))



@app.route('/terms')
def terms():
    return render_template('Terms.html')

@app.route('/privacy_policy')
def privacy():
    return render_template('PrivacyPolicy.html')

@app.route('/AboutUsMain')
def about():
    return render_template('AboutUsMain.html')

@app.route('/AboutUsIn')
def about1():
    return render_template('AboutUsIn.html')

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's port or default to 5000
    app.run(host="0.0.0.0", port=port)