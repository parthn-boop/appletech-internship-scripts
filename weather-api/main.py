from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
import requests

app = FastAPI(
    title="Global Weather API",
    description="Get current and historical weather data by city and date.",
    version="1.0.0"
)

# Centralized exception handler for unexpected errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Unexpected server error"}
    )

# Function for latest weather
def get_latest_weather(city: str):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_response = requests.get(geo_url)

        if geo_response.status_code == 429:
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded"})
        if geo_response.status_code >= 500:
            raise HTTPException(status_code=502, detail={"error": "Weather provider unavailable"})

        geo_data = geo_response.json()
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            raise HTTPException(status_code=404, detail={"error": "Please enter correct city name"})

        latitude = geo_data["results"][0]["latitude"]
        longitude = geo_data["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&current_weather=true"
            f"&hourly=relative_humidity_2m,precipitation"
            f"&daily=sunrise,sunset"
            f"&timezone=auto"
        )
        weather_response = requests.get(weather_url)

        if weather_response.status_code == 429:
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded"})
        if weather_response.status_code >= 500:
            raise HTTPException(status_code=502, detail={"error": "Weather provider unavailable"})

        data = weather_response.json()
        return {
            "city": city,
            "temperature": data["current_weather"]["temperature"],
            "windspeed": data["current_weather"]["windspeed"],
            "humidity": data["hourly"]["relative_humidity_2m"][0],
            "rainfall": data["hourly"]["precipitation"][0],
            "sunrise": data["daily"]["sunrise"][0],
            "sunset": data["daily"]["sunset"][0]
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail={"error": "Weather service timeout"})


# Function for historical weather
def get_historical_weather(city: str, date: str):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_response = requests.get(geo_url)

        if geo_response.status_code == 429:
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded"})
        if geo_response.status_code >= 500:
            raise HTTPException(status_code=502, detail={"error": "Weather provider unavailable"})

        geo_data = geo_response.json()
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            raise HTTPException(status_code=404, detail={"error": "City not found"})

        latitude = geo_data["results"][0]["latitude"]
        longitude = geo_data["results"][0]["longitude"]

        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={date}&end_date={date}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
            f"&daily=sunrise,sunset"
            f"&timezone=auto"
        )
        weather_response = requests.get(weather_url)

        if weather_response.status_code == 429:
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded"})
        if weather_response.status_code >= 500:
            raise HTTPException(status_code=502, detail={"error": "Weather provider unavailable"})

        data = weather_response.json()
        return {
            "city": city,
            "date": date,
            "temperature": data["hourly"]["temperature_2m"][0],
            "humidity": data["hourly"]["relative_humidity_2m"][0],
            "rainfall": data["hourly"]["precipitation"][0],
            "sunrise": data["daily"]["sunrise"][0],
            "sunset": data["daily"]["sunset"][0]
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail={"error": "Weather service timeout"})


# Endpoint 1: Latest weather
@app.get("/weather/latest", summary="Get latest weather by city")
def weather_latest(city: str = Query(..., description="City name, e.g. 'London'")):
    if not city:
        raise HTTPException(status_code=400, detail={"error": "City parameter is required"})
    return get_latest_weather(city)


# Endpoint 2: Historical weather
@app.get("/weather/history", summary="Get historical weather by city and date")
def weather_history(
    city: str = Query(..., description="City name, e.g. 'Paris'"),
    date: str = Query(..., description="Date in YYYY-MM-DD format, e.g. '2026-05-10'")
):
    if not city:
        raise HTTPException(status_code=400, detail={"error": "City parameter is required"})
    if not date:
        raise HTTPException(status_code=400, detail={"error": "Date parameter is required"})
    return get_historical_weather(city, date)
