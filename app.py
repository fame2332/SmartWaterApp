import pickle
import numpy as np
from flask import Flask, jsonify, render_template, flash, redirect, url_for, request, session, Response, send_file
from flask_apscheduler import APScheduler
from flask_cors import CORS
import requests
import pymysql.cursors
from werkzeug.security import generate_password_hash, check_password_hash
import math
from collections import defaultdict, Counter
from datetime import datetime, date, timedelta
import os
from flask_mail import Mail, Message
import csv
import io


app = Flask(__name__)


app.config['MAIL_SERVER']= 'smtp.gmail.com'
app.config['MAIL_PORT']= 465
app.config['MAIL_USERNAME']= 'smartwater696@gmail.com'
app.config['MAIL_PASSWORD']= 'ypnv smsg tsth fbdd'
app.config['MAIL_USE_TLS']= False
app.config['MAIL_USE_SSL']= True
mail = Mail(app)


class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


def fetch_and_process_alerts():
    forecast_url = f"{WEATHER_BASE_URL}/forecast?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"
    response = requests.get(forecast_url)

    if response.status_code != 200:
        print("Error: Unable to fetch forecast data")
        return

    data = response.json()

    if data['cod'] != '200':
        print("Error: Invalid forecast response")
        return

    # Get today's date
    today_date = date.today().strftime('%Y-%m-%d')

    # Initialize alerts and track times
    rain_alert = False
    fog_alert = False
    highest_fog_likelihood = 0
    rain_times = []
    fog_times = []

    # Process today's forecast entries
    for item in data['list']:
        forecast_date, forecast_time = item['dt_txt'].split(' ')
        if forecast_date != today_date:
            continue  # Skip non-today forecasts

        # Check for rain
        rain_mm = item.get('rain', {}).get('3h', 0)  # Rain in mm over 3 hours
        if rain_mm > 0:
            rain_alert = True
            rain_times.append(forecast_time)  # Track time of rain

        # Prepare features for fog likelihood prediction
        features = [
            item['main']['temp'],
            calculate_dew_point(item['main']['temp'], item['main']['humidity']),
            item['main']['humidity'],
            item['wind']['speed'],
            item.get('visibility', 10000) / 1000,  # Default to 10 km if visibility data is missing
            item['clouds']['all'],  # Cloud cover in %
            item['main']['feels_like']
        ]

        # Scale features and predict fog likelihood
        scaled_features = scaler.transform([features])
        fog_probability = fog_model.predict_proba(scaled_features)[0][1]
        fog_likelihood_percentage = round(fog_probability * 100, 2)

        # Update fog alert if likelihood is above threshold
        if fog_likelihood_percentage >= 30:
            fog_alert = True
            fog_times.append(forecast_time)  # Track time of fog

        # Track highest fog likelihood
        highest_fog_likelihood = max(highest_fog_likelihood, fog_likelihood_percentage)

    # Send email alerts if conditions are met
    if rain_alert or fog_alert:
        send_alert_emails(rain_alert, rain_times, fog_alert, fog_times, highest_fog_likelihood)

    print({
        'rain_alert': rain_alert,
        'fog_alert': fog_alert,
        'highest_fog_likelihood': highest_fog_likelihood,
        'rain_times': rain_times,
        'fog_times': fog_times
    })


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




BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_BASE_URL = 'http://api.openweathermap.org/data/2.5'
API_KEY = '0441c36366b282b3fdbafb63653ca358'
CITY = 'Bacoor'
COUNTRY = 'PH'
AREA_M2 = 50  # Area for rainwater harvesting in square meters
EFFICIENCY_COEFFICIENT = 0.8  # Efficiency for rainwater harvesting
with open('fog_prediction_model.pkl', 'rb') as model_file:
    loaded_data = pickle.load(model_file)
    if isinstance(loaded_data, dict):
        fog_model = loaded_data['model']
        scaler = loaded_data['scaler']
        optimal_threshold = loaded_data['threshold']
    else:
        raise ValueError("Loaded data is not in the expected format. Ensure the model was saved as a dictionary.")

def calculate_dew_point(T, RH):

    # Constants for Magnus formula
    a = 17.27
    b = 237.7

    gamma = (a * T / (b + T)) + math.log(RH / 100.0)

    dew_point = (b * gamma) / (a - gamma)

    return round(dew_point, 2)


@app.route('/table_data', methods=['GET'])
def table_data():
    city = "Bacoor,PH"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return {"error": "Failed to fetch weather data"}, response.status_code

    data = response.json()

    # Load fog prediction model
    with open('fog_prediction_model.pkl', 'rb') as model_file:
        loaded_data = pickle.load(model_file)
        if isinstance(loaded_data, dict):
            fog_model = loaded_data['model']
            scaler = loaded_data['scaler']
        else:
            return jsonify({"error": "Fog model is not in the expected format."})

    forecast_list = []
    for forecast in data['list']:
        dt_txt = forecast['dt_txt']
        date, time = dt_txt.split(" ")
        temp = forecast['main']['temp']
        feels_like = forecast['main']['feels_like']
        humidity = forecast['main']['humidity']
        wind_speed = forecast['wind']['speed']
        pressure = forecast['main']['pressure']
        visibility = forecast.get('visibility', 10000) / 1000  # Convert to km
        cloud_cover = forecast.get('clouds', {}).get('all', 0)
        rain_3hr = forecast.get('rain', {}).get('3h', 0)
        rain_1hr = forecast.get('rain', {}).get('1h', 0)
        dew_point = temp - ((100 - humidity) / 5)  # Simple dew point formula
        rain_liters = 0.8 * 50 * (rain_3hr / 1000) * 1000
        features = scaler.transform([[temp, dew_point, humidity, wind_speed, visibility, cloud_cover, feels_like]])
        fog_prob = fog_model.predict_proba(features)[:, 1][0] * 100

        forecast_list.append({
            "date": date,
            "time": time,
            "temperature": round(temp, 2),
            "feelsLike": round(feels_like, 2),
            "humidity": round(humidity, 2),
            "rain3hr": round(rain_3hr, 2),
            "rain1hr": round(rain_1hr, 2),
            "windSpeed": round(wind_speed, 2),
            "dewPoint": round(dew_point, 2),
            "rainfall": round(rain_liters, 2),
            "fogLikelihood": round(fog_prob, 2),
            "pressure": round(pressure, 2),
            "visibility": round(visibility, 2),
            "cloudCover": round(cloud_cover, 2)
        })

    return jsonify(forecast_list)

@app.route('/download_csv', methods=['GET'])
def download_csv():
    API_KEY = "NJEU4MUZCARCCSTS7SV8HL6VE"
    BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    LOCATION = "Bacoor, Cavite, Philippines"
    CSV_FILE = "weather_data.csv"

    # Load the fog prediction model
    try:
        with open('fog_prediction_model.pkl', 'rb') as model_file:
            loaded_data = pickle.load(model_file)
            if isinstance(loaded_data, dict):
                fog_model = loaded_data['model']
                scaler = loaded_data['scaler']
                optimal_threshold = loaded_data['threshold']
            else:
                raise ValueError("Loaded data is not in the expected format. Ensure the model was saved as a dictionary.")
    except Exception as e:
        return jsonify({"error": f"Error loading fog model: {str(e)}"})

    try:
        # Calculate the date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=2)

        # Fetch data from the Visual Crossing API
        url = f"{BASE_URL}/{LOCATION}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?unitGroup=metric&include=hours&key={API_KEY}&contentType=json"
        response = requests.get(url)

        if response.status_code != 200:
            return jsonify({"error": f"Error fetching data: {response.status_code}, {response.text}"}), response.status_code

        weather_data = response.json().get('days', [])

        # Create CSV file
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['datetime', 'temp', 'feelslike', 'humidity', 'dew', 'precip', 'precipprob', 'windspeed', 'visibility', 'windcover', 'foglikelihood (%)', 'harvest_liters'])

            AREA_M2 = 50  # Area for rainwater harvesting in square meters
            EFFICIENCY_COEFFICIENT = 0.8  # Efficiency for rainwater harvesting

            for day in weather_data:
                for hour in day.get('hours', []):
                    datetime_str = f"{day['datetime']}:{hour.get('datetime')}"  # Format: 2024-11-22:6:00:00
                    temp = hour.get('temp')
                    feelslike = hour.get('feelslike')
                    humidity = hour.get('humidity')
                    dew = hour.get('dew')
                    precip = hour.get('precip', 0)  # Default to 0 if missing
                    precipprob = hour.get('precipprob', 0)  # Probability of precipitation
                    windspeed = hour.get('windspeed')
                    visibility = hour.get('visibility')
                    windcover = hour.get('cloudcover')  # Assuming cloudcover is equivalent to windcover

                    # Calculate fog likelihood as a percentage
                    features = scaler.transform([[temp, feelslike, humidity, dew, precip, windspeed, visibility]])
                    fog_probability = fog_model.predict_proba(features)[0][1]
                    foglikelihood_percentage = round(fog_probability * 100, 2)

                    # Calculate harvest liters
                    harvest_liters = precip * AREA_M2 * EFFICIENCY_COEFFICIENT

                    writer.writerow([
                        datetime_str,
                        temp,
                        feelslike,
                        humidity,
                        dew,
                        precip,
                        precipprob,
                        windspeed,
                        visibility,
                        windcover,
                        foglikelihood_percentage,
                        harvest_liters
                    ])

        # Return the file for download
        return send_file(CSV_FILE, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/')
def home():
    return render_template('landingpage.html')

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/index_graph')
def index_graph():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard_graph.html')

@app.route('/index_table')
def index_table():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard_table.html')

@app.route('/index_data', methods=['GET'])
def index_data():
    current_weather_url = f"{WEATHER_BASE_URL}/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"
    forecast_url = f"{WEATHER_BASE_URL}/forecast?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"

    try:
        # Fetch current weather data
        current_response = requests.get(current_weather_url)
        forecast_response = requests.get(forecast_url)

        if current_response.status_code == 200 and forecast_response.status_code == 200:
            current_data = current_response.json()
            forecast_data = forecast_response.json()

            if forecast_data['cod'] != '200':
                return jsonify({'error': forecast_data.get('message', 'Invalid forecast response')}), 400

            # Initialize variables for forecast rainfall sum
            total_rainfall_mm = 0
            area_m2 = 50  # Bacoor area in square meters
            efficiency_coefficient = 0.8  # Efficiency for rainwater harvesting

            today_date = date.today().strftime('%Y-%m-%d')

            # Sum up rainfall for today's forecasted periods
            for item in forecast_data['list']:
                forecast_date, _ = item['dt_txt'].split(' ')
                if forecast_date == today_date:
                    rainfall_mm = item.get('rain', {}).get('3h', 0)  # Rainfall in mm over 3 hours
                    total_rainfall_mm += rainfall_mm

            # Calculate total liters of rain harvested
            total_liters_of_rain = total_rainfall_mm * area_m2 * efficiency_coefficient

            # Extract current weather data
            temperature = current_data['main']['temp']
            humidity = current_data['main']['humidity']
            rainfall_mm = current_data.get('rain', {}).get('1h', 0) or current_data.get('rain', {}).get('3h', 0)
            description = current_data['weather'][0].get('description', "N/A")
            windspeed = current_data['wind']['speed']
            visibility = current_data.get('visibility', 10000) / 1000  # Convert visibility to km
            cloudcover = current_data['clouds']['all']
            feelslike = current_data['main']['feels_like']

            # Calculate dew point
            dew_point = calculate_dew_point(temperature, humidity)

            # Prepare features for fog likelihood prediction
            features = np.array([[temperature, dew_point, humidity, windspeed, visibility, cloudcover, feelslike]])

            # Scale features and predict fog likelihood
            scaled_features = scaler.transform(features)
            prediction_prob = fog_model.predict_proba(scaled_features)[0][1]
            fog_likelihood_percentage = round(prediction_prob * 100, 2)

            # Return the response
            return jsonify({
                "temperature": f"{temperature}°C",
                "humidity": humidity,
                "rainfall_mm": rainfall_mm,  # Rainfall for the current time
                "description": description,
                "dew_point": dew_point,
                "harvest": round(total_liters_of_rain, 2),  # Total liters harvested
                "fogLikelihood": fog_likelihood_percentage  # Fog likelihood percentage
            })

        else:
            # Return an error message if any API request fails
            return jsonify({'error': 'Could not fetch weather data'}), max(current_response.status_code, forecast_response.status_code)

    except Exception as e:
        # Return an error message if something goes wrong with the request
        return jsonify({'error': str(e)}), 500




@app.route('/download-weather-report', methods=['GET'])
def download_weather_report():
    # API parameters
    city = "Bacoor,PH"  # Replace with your city
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # Temperature in Celsius
    }

    # Fetch weather data
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return {"error": "Failed to fetch weather data"}, response.status_code

    data = response.json()

    # Prepare CSV file
    csv_output = io.StringIO()
    writer = csv.writer(csv_output)
    writer.writerow([
        "Date", "Time", "Temperature (°C)", "Feels Like (°C)",
        "Humidity (%)", "Rain (mm/3hr)", "Rain (mm/1hr)",
        "Wind Speed (m/s)", "Dew Point (°C)", "Rainfall (liters)",
        "Fog Likelihood (%)", "Pressure (hPa)", "Visibility (km)", "Cloud Cover (%)"
    ])

    # Load fog prediction model
    with open('fog_prediction_model.pkl', 'rb') as model_file:
        loaded_data = pickle.load(model_file)
        if isinstance(loaded_data, dict):
            fog_model = loaded_data['model']
            scaler = loaded_data['scaler']
            optimal_threshold = loaded_data['threshold']
        else:
            raise ValueError("Loaded data is not in the expected format. Ensure the model was saved as a dictionary.")

    # Constants for rainfall calculation
    area_m2 = 50  # Fixed area in m²
    coefficient = 0.8  # Harvesting coefficient

    # Process each forecast entry
    for forecast in data['list']:
        dt_txt = forecast['dt_txt']
        date, time = dt_txt.split(" ")
        temp = forecast['main']['temp']
        feels_like = forecast['main']['feels_like']
        humidity = forecast['main']['humidity']
        wind_speed = forecast['wind']['speed']
        pressure = forecast['main']['pressure']  # Pressure in hPa
        visibility = forecast.get('visibility', 10000) / 1000  # Visibility in km
        cloud_cover = forecast.get('clouds', {}).get('all', 0)  # Cloud cover in %

        rain_3hr = forecast.get('rain', {}).get('3h', 0)  # Rain in mm/3hr
        rain_1hr = forecast.get('rain', {}).get('1h', 0)  # Rain in mm/1hr
        dew_point = calculate_dew_point(temp, humidity)

        # Calculate rainfall in liters
        rain_liters = coefficient * area_m2 * (rain_3hr / 1000) * 1000  # Convert mm to meters

        # Prepare features for fog prediction
        forecast_features = [
            temp, dew_point, humidity, wind_speed, visibility,
            cloud_cover, feels_like
        ]
        scaled_features = scaler.transform([forecast_features])
        fog_prob = fog_model.predict_proba(scaled_features)[:, 1][0] * 100  # Convert to percentage

        # Write data to CSV
        writer.writerow([
            date, time, temp, feels_like, humidity, rain_3hr, rain_1hr,
            wind_speed, dew_point, round(rain_liters, 2), round(fog_prob, 2),
            pressure, round(visibility, 2), cloud_cover
        ])

    # Send CSV as a downloadable file
    output = Response(csv_output.getvalue(), mimetype="text/csv")
    output.headers["Content-Disposition"] = "attachment; filename=weather_report.csv"
    return output



@app.route('/forecast', methods=['GET'])
def get_forecast():
    params = {
        'q': f"{CITY},{COUNTRY}",
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch forecast data'}), 500

    data = response.json()
    forecast_list = data.get('list', [])
    result = []

    # Prepare datasets for graphs
    chart1_data = {
        "labels": [],
        "datasets": {
            "temperature": [],
            "dewPoint": [],
            "feelsLike": [],
            "humidity": []
        }
    }
    chart2_data = {
        "labels": [],
        "datasets": {
            "rainLiters": [],
            "fogLikelihood": []
        }
    }
    chart3_data = {
        "labels": [],
        "datasets": {
            "windSpeed": []
        }
    }

    for entry in forecast_list:
        dt = datetime.utcfromtimestamp(entry['dt']).strftime('%Y-%m-%d %H:%M:%S')
        time_only = datetime.utcfromtimestamp(entry['dt']).strftime('%H:%M')

        temperature = entry['main']['temp']
        humidity = entry['main']['humidity']
        wind_speed = entry['wind']['speed']
        rainfall_3hr = entry.get('rain', {}).get('3h', 0.0)
        rainfall_1hr = rainfall_3hr / 3
        dew_point = calculate_dew_point(temperature, humidity)
        visibility = entry.get('visibility', 10000) / 1000  # Convert meters to kilometers
        cloud_cover = entry.get('clouds', {}).get('all', 0)
        feels_like = entry['main']['feels_like']
        harvest = rainfall_3hr * AREA_M2 * EFFICIENCY_COEFFICIENT

        # Prepare features for fog likelihood prediction
        feature_values = [[
            temperature, dew_point, humidity, wind_speed, visibility, cloud_cover, feels_like
        ]]
        scaled_features = scaler.transform(feature_values)
        fog_probability = fog_model.predict_proba(scaled_features)[0][1] * 100

        # Add data to result
        result.append({
            'dateWithHour': dt,
            'temperature': round(temperature, 2),
            'windSpeed': round(wind_speed, 2),
            'humidity': round(humidity, 2),
            'rainfall3hrs_mm': round(rainfall_3hr, 2),
            'rainfall1hr_mm': round(rainfall_1hr, 2),
            'rainfallLiters': round(harvest, 2),
            'dewPoint': dew_point,
            'harvest': round(harvest, 2),
            'fogLikelihood': round(fog_probability, 2),  # Percentage
            'feelsLike': round(feels_like, 2),
            'visibility_km': round(visibility, 2),
            'cloudCover': round(cloud_cover, 2)
        })

        # Populate chart data
        chart1_data["labels"].append(time_only)
        chart1_data["datasets"]["temperature"].append(round(temperature, 2))
        chart1_data["datasets"]["dewPoint"].append(dew_point)
        chart1_data["datasets"]["feelsLike"].append(round(feels_like, 2))
        chart1_data["datasets"]["humidity"].append(round(humidity, 2))

        chart2_data["labels"].append(dt)
        chart2_data["datasets"]["rainLiters"].append(round(harvest, 2))
        chart2_data["datasets"]["fogLikelihood"].append(round(fog_probability, 2))

        chart3_data["labels"].append(dt)
        chart3_data["datasets"]["windSpeed"].append(round(wind_speed, 2))

    return jsonify({
        "forecast": result,
        "chart1": chart1_data,
        "chart2": chart2_data,
        "chart3": chart3_data
    })


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

            # Process daily data
            processed_forecast = []
            today = date.today()

            for forecast_date, values in sorted(daily_data.items()):
                # Calculate average temperature
                avg_temp = round(sum(temp for temp, _, _ in values) / len(values))  # Average temperature

                # Calculate total rainfall for the day
                total_rainfall_mm = sum(rainfall for _, _, rainfall in values if rainfall > 0)

                # Calculate harvest (liters of rain)
                area_m2 = 50  # Assuming a fixed area of 50 m²
                liters_of_rain = total_rainfall_mm * area_m2 * 0.8  # 80% efficiency for rainwater harvesting
                harvest = round(liters_of_rain, 2)


                description_counts = Counter(desc for _, desc, _ in values)
                descriptions_sorted = description_counts.most_common()

                most_common = descriptions_sorted[0][0].capitalize() if descriptions_sorted else "No significant data"
                second_common = None

                max_count = description_counts[most_common]


                for i in range(len(descriptions_sorted)):
                    for j in range(i + 1, len(descriptions_sorted)):
                        desc1 = descriptions_sorted[i][0]
                        desc2 = descriptions_sorted[j][0]
                        combined_count = description_counts[desc1] + description_counts[desc2]

                        if combined_count > max_count:
                            max_count = combined_count
                            most_common = desc1.capitalize()
                            second_common = desc2.capitalize()

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
                    "total_rainfall_mm": total_rainfall_mm,
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
            if key == 'temp':
                return {"temperature": value}
            elif key == 'humidity':
                return {"humidity": value}
            elif key == 'wind_speed':
                return {"windspeed": value}
            elif key == 'visibility':
                return {"visibility": value / 1000}  # Convert to km
            elif key == 'cloudcover':
                return {"cloudcover": value}
            elif key == 'feelslike':
                return {"feelslike": value}
            elif key == 'rain_mm':
                return {"rain_mm": value}
            else:
                return {}

        mid = len(data) // 2
        left = dict(list(data.items())[:mid])
        right = dict(list(data.items())[mid:])
        left_result = divide_and_conquer_weather(left)
        right_result = divide_and_conquer_weather(right)
        return {**left_result, **right_result}

    # Fetch forecast weather data
    forecast_url = f"{WEATHER_BASE_URL}/forecast?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"
    response = requests.get(forecast_url)

    if response.status_code != 200:
        return jsonify({"error": "Unable to fetch forecast data"}), 400

    forecast_data = response.json()

    # Get the current date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')

    rain_alert = False
    fog_alert = False

    # Filter forecasts for the current day only
    today_forecasts = [
        forecast for forecast in forecast_data['list']
        if forecast['dt_txt'].startswith(today_date)
    ]

    # Iterate through today's forecast data (3-hour intervals)
    for forecast in today_forecasts:
        # Extract relevant features for each forecast period
        raw_features = {
            "temp": forecast['main']['temp'],
            "humidity": forecast['main']['humidity'],
            "wind_speed": forecast['wind']['speed'],
            "visibility": forecast.get('visibility', 10000),  # Default to max visibility if not present
            "cloudcover": forecast['clouds']['all'],
            "feelslike": forecast['main']['feels_like']
        }

        # Check for rain data
        rain_mm = forecast.get('rain', {}).get('3h', 0)
        raw_features['rain_mm'] = rain_mm

        # Process features using divide and conquer
        processed_features = divide_and_conquer_weather(raw_features)

        # Calculate dew point
        processed_features['dew_point'] = calculate_dew_point(
            processed_features['temperature'], processed_features['humidity']
        )

        # Scale features for prediction
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
        fog_likelihood_percentage = round(prediction_prob * 100, 2)

        # Update alerts based on thresholds
        if rain_mm > 0:
            rain_alert = True
        if fog_likelihood_percentage >= 30:
            fog_alert = True

    # If no alerts are active, return a message
    if not rain_alert and not fog_alert:
        return jsonify({
            'message': "No alerts for the forecasted periods.",
            'rain_alert': rain_alert,
            'fog_alert': fog_alert
        })

    # Return the response with the alerts
    return jsonify({
        'rain_alert': rain_alert,
        'fog_alert': fog_alert,
        'message': "Weather alerts active for the forecasted periods."
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
                    session['username'] = username  # Store username in session
                    session['email'] = user['email']  # Store user email in session
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


@app.route('/alerts', methods=['GET'])
def get_alerts():
    fetch_and_process_alerts()
    return jsonify({"message": "Alerts processed manually."})

# Scheduler Job
@scheduler.task('interval', id='fetch_alerts_job', hours=1)
def scheduled_alert_job():
    fetch_and_process_alerts()

def send_alert_emails(rain_alert, rain_times, fog_alert, fog_times, fog_likelihood_percentage):
    try:
        with db.cursor() as cursor:
            # Fetch all registered email addresses
            cursor.execute("SELECT email FROM users")  # Replace 'users' with your table name
            emails = [row['email'] for row in cursor.fetchall()]

        # Compose the email subject and message
        subject = "Weather Alert: Rain and/or Fog Expected"
        body = "Weather Alert!\n\n"

        if rain_alert:
            body += "Rain is forecasted in your area at the following times:\n"
            body += ", ".join(rain_times) + "\n"
            body += "Please set up a rainwater collection system today.\n\n"
        if fog_alert:
            body += "Fog is likely at the following times:\n"
            body += ", ".join(fog_times) + "\n"
            body += f"Fog likelihood is forecasted to be as high as {fog_likelihood_percentage}%. Please set up a fog net in your area today.\n\n"

        body += "Stay safe!\nYour Smart Water Team"

        # Send emails to all registered users
        with mail.connect() as conn:
            for email in emails:
                msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[email])
                msg.body = body
                conn.send(msg)
    except Exception as e:
        print(f"Error sending email: {e}")








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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)