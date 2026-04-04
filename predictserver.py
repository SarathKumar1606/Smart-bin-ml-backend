from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta
import holidays
import pytz

# ==========================================================
# APP INITIALIZATION
# ==========================================================

app = Flask(__name__)

wet_model = joblib.load("wet_model.pkl")
dry_model = joblib.load("dry_model.pkl")

# Indian Standard Time
IST = pytz.timezone("Asia/Kolkata")
india_holidays = holidays.India(years=range(2024, 2030))

# ==========================================================
# CONFIGURATION
# ==========================================================

WET_THRESHOLD = 90
DRY_THRESHOLD = 90
ALERT_LIMIT_HOURS = 2

# ==========================================================
# HOLIDAY CLASSIFICATION ENGINE
# ==========================================================

def get_holiday_factor(current_date):
    holiday_name = india_holidays.get(current_date)

    if not holiday_name:
        return 0.0, 0, None

    holiday_name_clean = holiday_name.split(",")[0].strip()

    MAJOR = ["Diwali", "Holi", "Eid", "Christmas"]
    NATIONAL = ["Republic Day", "Independence Day", "Gandhi Jayanti"]

    for k in MAJOR:
        if k.lower() in holiday_name_clean.lower():
            return 0.7, 1, holiday_name_clean

    for k in NATIONAL:
        if k.lower() in holiday_name_clean.lower():
            return 0.3, 1, holiday_name_clean

    return 0.25, 1, holiday_name_clean


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def home():
    return "Smart Dustbin ML Backend Running"

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # Inputs
        wet_level = float(data.get("wet_level", 0))
        dry_level = float(data.get("dry_level", 0))
        avg_fill_rate_last_3h = float(data.get("avg_fill_rate_last_3h", 0))
        previous_day_same_time_level = float(
            data.get("previous_day_same_time_level", 0)
        )
        weather_condition = str(data.get("weather_condition", "normal"))

        # Time
        now = datetime.now(IST)

        # Holiday
        holiday_factor, is_holiday, holiday_name = get_holiday_factor(now.date())

        # DataFrame
        input_df = pd.DataFrame([{
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "holiday_factor": float(holiday_factor),
            "is_holiday": int(is_holiday),
            "weather_condition": weather_condition,
            "wet_level": wet_level,
            "dry_level": dry_level,
            "avg_fill_rate_last_3h": avg_fill_rate_last_3h,
            "previous_day_same_time_level": previous_day_same_time_level
        }])

        # Predictions
        wet_rate = max(float(wet_model.predict(input_df)[0]), 0.01)
        dry_rate = max(float(dry_model.predict(input_df)[0]), 0.01)

        wet_hours = max((WET_THRESHOLD - wet_level) / wet_rate, 0)
        dry_hours = max((DRY_THRESHOLD - dry_level) / dry_rate, 0)

        if wet_hours <= dry_hours:
            selected_bin = "wet"
            final_hours = wet_hours
        else:
            selected_bin = "dry"
            final_hours = dry_hours

        pickup_time = now + timedelta(hours=float(final_hours))

        return jsonify({
            "selected_bin_for_pickup": selected_bin,
            "wet_hours_remaining": wet_hours,
            "dry_hours_remaining": dry_hours,
            "final_hours_remaining": final_hours,
            "next_pickup_datetime": pickup_time.strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_required_immediately": final_hours <= ALERT_LIMIT_HOURS,
            "is_holiday_today": bool(is_holiday),
            "holiday_name": holiday_name,
            "current_time": now.strftime("%H:%M:%S")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
