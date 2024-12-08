document.addEventListener('DOMContentLoaded', async () => {
  // Fetch the current day and date from the Flask API
  const fetchCurrentDate = async () => {
    try {
      const response = await fetch('/current_date');
      if (response.ok) {
        return await response.json(); // Parse JSON response
      } else {
        console.error('Failed to fetch current date and day');
        return null;
      }
    } catch (error) {
      console.error('Error:', error);
      return null;
    }
  };

  // Function to update the day and date display
  const updateDateAndDay = (data) => {
    if (data) {
      document.querySelector('.day').textContent = data.day;
      document.querySelector('.full-date').textContent = data.full_date;
    } else {
      console.error('Date and day data is unavailable');
    }
  };

  // Fetch and update the date and day
  const currentDate = await fetchCurrentDate();
  updateDateAndDay(currentDate);

  // Fetch the weather data from Flask
  const fetchWeatherData = async () => {
    try {
      const response = await fetch('/index_data');
      if (response.ok) {
        return await response.json();
      } else {
        console.error('Failed to fetch weather data');
        return null;
      }
    } catch (error) {
      console.error('Error:', error);
      return null;
    }
  };

  const currentWeatherData = await fetchWeatherData();
  if (currentWeatherData) {
    const rainfallValueElement = document.querySelector('.depth-value');
    const rainfallValue = currentWeatherData.rainfall_mm;
    rainfallValueElement.textContent = rainfallValue;
  } else {
    console.error('Weather data is unavailable');
  }

function toggleWeatherBadge(condition, fogLikelihood) {
  const rainBadge = document.querySelector('.badge');
  const lowerCondition = condition.toLowerCase();  // Make sure the condition is in lowercase
  const fogLikelihoodPercentage = parseFloat(fogLikelihood);  // Ensure fogLikelihood is a valid number

  if (lowerCondition.includes('rain') || fogLikelihoodPercentage >= 33) {
    rainBadge.style.display = 'inline-block';  // Show badge
  } else {
    rainBadge.style.display = 'none';  // Hide badge
  }
}


  function updateWeatherInfo(data) {
    const tempElement = document.querySelector('.temp');
    const conditionElement = document.querySelector('.condition');
    const weatherIconElement = document.querySelector('.weather-icon');
    const humidityElement = document.querySelector('.humidity-value .value');
    const dewpointElement = document.querySelector('.dew-point-value .value');
    const harvestElement = document.querySelector('.harvested-value .value');
    const fogLikelihoodElement = document.querySelector('.fog-value .value');

    const temp = data?.temperature || 'N/A';
    const condition = data?.description || 'N/A';
    const humidity = data?.humidity || 'N/A';
    const dewpoint = data?.dew_point || 'N/A';
    const harvest = data?.harvest || '0';
    const fogLikelihood = data?.fogLikelihood || 'N/A';

    tempElement.textContent = temp;
    conditionElement.textContent = condition;
    humidityElement.textContent = humidity;
    dewpointElement.textContent = dewpoint;
    harvestElement.textContent = harvest;
    fogLikelihoodElement.textContent = fogLikelihood;

    const iconPath = getWeatherIcon(condition);
    weatherIconElement.src = iconPath;
    weatherIconElement.alt = condition;

    toggleWeatherBadge(condition, fogLikelihood);
  }

  if (currentWeatherData) {
    updateWeatherInfo(currentWeatherData);
  } else {
    console.error('Weather data is unavailable');
  }

  // Fetch the forecast data dynamically from Flask
  const fetchForecastData = async () => {
    try {
      const response = await fetch('/forecast_data');
      if (response.ok) {
        return await response.json(); // Parse the JSON response
      } else {
        console.error('Failed to fetch forecast data');
        return [];
      }
    } catch (error) {
      console.error('Error fetching forecast data:', error);
      return [];
    }
  };

  // Function to update the forecast cards
  const updateForecastCards = async () => {
    const forecastCardsContainer = document.getElementById('forecast-cards');
    const forecastData = await fetchForecastData();

    if (forecastData.length === 0) {
      console.error('No forecast data available');
      return;
    }

    forecastData.forEach((forecast) => {
      const card = document.createElement('div');
      card.classList.add('forecast-card');

      const weatherIcon = getWeatherIcon(forecast.condition);

      card.innerHTML = `
        <div class="left">
          <img src="${weatherIcon}" alt="Weather Icon" class="weather-icon">
          <div class="details">
            <span>${forecast.day} | ${forecast.date}</span>
            <span>${forecast.condition}</span>
          </div>
        </div>
        <div class="temperature">${forecast.temp}</div>
      `;

      forecastCardsContainer.appendChild(card);
    });
  };

  function getWeatherIcon(condition) {
    const lowerCondition = condition.toLowerCase();

    if (lowerCondition.includes('rain')) {
      return '/static/Images/rainy-1.gif';
    } else if (lowerCondition.includes('cloud')) {
      return '/static/Images/cloud-1.gif';
    } else if (lowerCondition.includes('clear')) {
      return '/static/Images/clear-1.gif';
    } else {
      return '/static/Images/default-icon.gif';
    }
  }

  // Update forecast cards
  await updateForecastCards();

  // Fetch the forecast data again for updating the chart
  const forecastData = await fetchForecastData();

  // Check if "rain" is in the condition of any forecast for today
  const hasRainToday = forecastData.some(forecast =>
    forecast.condition.toLowerCase().includes('rain')
  );

  // Show the badge if rain is detected for today
  const rainBadge = document.querySelector('.badge');
  if (hasRainToday) {
    rainBadge.style.display = 'inline-block';
  } else {
    rainBadge.style.display = 'none';
  }


  console.log(forecastData);
  // Chart Data
  const ctx = document.getElementById('barChart').getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 400);
  gradient.addColorStop(0, 'rgba(54, 162, 235, 0.8)');
  gradient.addColorStop(1, 'rgba(54, 162, 235, 0.2)');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: forecastData.map(item => item.day), // Days of the forecast
      datasets: [{
        label: 'Liters of Water',
        data: forecastData.map(item => item.harvest_liters), // Harvest liters from forecast data
        backgroundColor: gradient,
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          },
          ticks: {
            color: '#1A2B4',
            font: {
              size: 14
            }
          }
        },
        y: {
          grid: {
            color: 'rgba(200, 200, 200, 0.2)',
          },
          ticks: {
            beginAtZero: true,
            color: '#1A2B4',
            font: {
              size: 14
            }
          }
        }
      }
    }
  });
});

// URL of the backend endpoint
const apiUrl = '/wind';

// Wind direction and speed update logic
const windDirections = {
    'N': { min: 350, max: 10 },
    'N/NE': { min: 20, max: 30 },
    'NE': { min: 40, max: 50 },
    'E/NE': { min: 60, max: 70 },
    'E': { min: 80, max: 100 },
    'E/SE': { min: 110, max: 120 },
    'SE': { min: 130, max: 140 },
    'S/SE': { min: 150, max: 160 },
    'S': { min: 170, max: 190 },
    'S/SW': { min: 200, max: 210 },
    'SW': { min: 220, max: 230 },
    'W/SW': { min: 240, max: 250 },
    'W': { min: 260, max: 280 },
    'W/NW': { min: 290, max: 300 },
    'NW': { min: 310, max: 320 },
    'N/NW': { min: 330, max: 340 }
};

// Function to determine wind direction based on degree
function getWindDirection(degree) {
    for (let direction in windDirections) {
        let range = windDirections[direction];
        if (degree >= range.min && degree <= range.max) {
            return direction;
        }
        if (range.length === 4 && (degree >= range[0] || degree <= range[3])) {
            return direction;
        }
    }
    return 'N';
}

// Select DOM elements to update
const arrow = document.querySelector('.compass-arrow');
const windSpeedValue = document.querySelector('.wind-speed .value');
const windSpeedUnit = document.querySelector('.wind-speed .unit');
const windDirectionLabelElement = document.querySelector('.wind-direction');

// Global windData object that will be updated with data from the backend
let windData = {
    direction: 222,  // default value, will be updated
    speed: 7         // default value, will be updated
};

// Function to update the UI with wind data
function updateWindData() {
    // Update the wind direction arrow based on the fetched direction
    arrow.style.transform = `translate(-50%, -50%) rotate(${windData.direction}deg)`;

    // Update the wind speed
    windSpeedValue.textContent = windData.speed;
    windSpeedUnit.textContent = 'm/s';

    // Get the wind direction label (N, NE, etc.) and display it
    const windDirectionLabel = getWindDirection(windData.direction);
    windDirectionLabelElement.textContent = `Wind Direction: ${windDirectionLabel}`;
}

// Fetch data from the backend
async function fetchWeatherData() {
    try {
        const response = await fetch(apiUrl);
        const data = await response.json();

        if (response.ok) {
            // Update the windData object with the values from the backend
            windData = {
                direction: data.wind_dir,  // Wind direction from the backend response
                speed: data.wind_speed     // Wind speed from the backend response
            };

            // Now that windData is updated, call the function to update the UI
            updateWindData();
        } else {
            console.error('Failed to fetch weather data:', data.error);
        }
    } catch (error) {
        console.error('Error fetching weather data:', error);
    }
}

// Fetch and update data when the page loads
fetchWeatherData();
document.addEventListener('DOMContentLoaded', function() {

    // Get modal, bell icon, and close button
    var modal = document.getElementById("notificationModal");
    var bellIcon = document.getElementById("bell-icon");  // Make sure the bell icon has the correct ID
    var closeBtn = document.getElementById("closeModal");

    // Get references to the alert elements
    var rainAlert = document.querySelector('.rain-alert');
    var fogAlert = document.querySelector('.fog-alert');
    var noAlertMessage = document.querySelector('.no-alert-message'); // Add a reference to the "no alert" message

    // Function to show or hide alerts based on the response data
    function showAlerts(data) {
        // Hide the "no alert" message by default
        noAlertMessage.style.display = 'none';

        // Show or hide the rain alert based on the flag
        if (data.rain_alert) {
            rainAlert.style.display = 'block';
        } else {
            rainAlert.style.display = 'none';
        }

        // Show or hide the fog alert based on the flag
        if (data.fog_alert) {
            fogAlert.style.display = 'block';
        } else {
            fogAlert.style.display = 'none';
        }

        // If both alerts are false, show "no alert" message
        if (!data.rain_alert && !data.fog_alert) {
            noAlertMessage.style.display = 'block';  // Display the "no alert" message
        }

        // Show the modal if there is an alert or "no alert" message
        modal.style.display = "block";
    }

    // When the user clicks on the bell icon, open the modal and fetch the weather data
    bellIcon.addEventListener("click", function() {
        fetch('/predict_fog')
            .then(response => response.json())
            .then(data => {
                showAlerts(data);  // Show the alerts based on the backend data
            })
            .catch(error => {
                console.error('Error fetching weather data:', error);
            });
    });

    // When the user clicks on the close button, close the modal
    closeBtn.addEventListener("click", function() {
        modal.style.display = "none";
    });

    // When the user clicks anywhere outside of the modal, close it
    window.addEventListener("click", function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    });

});
