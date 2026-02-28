from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta
import holidays
import os
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

MAJOR_FESTIVALS = [
    "Diwali", "Deepavali",
    "Holi",
    "Eid", "Bakrid",
    "Christmas"
]

NATIONAL_HOLIDAYS = [
    "Republic Day",
    "Independence Day",
    "Gandhi Jayanti"
]

MODERATE_FESTIVALS = [
    "Dussehra", "Vijayadashami",
    "Navratri",
    "Janmashtami",
    "Raksha Bandhan",
    "Ganesh Chaturthi",
    "Durga Puja",
    "Onam",
    "Pongal",
    "Baisakhi"
]

MINOR_RELIGIOUS = [
    "Mahavir",
    "Buddha",
    "Muharram",
    "Shivaratri",
    "Ram Navami",
    "Makar Sankranti"
]


def get_holiday_factor(current_date):
    holiday_name = india_holidays.get(current_date)

    if not holiday_name:
        return 0.0, 0, None

    holiday_name_clean = holiday_name.split(",")[0].strip()

    for keyword in MAJOR_FESTIVALS:
        if keyword.lower() in holiday_name_clean.lower():
            return 0.7, 1, holiday_name_clean

    for keyword in NATIONAL_HOLIDAYS:
        if keyword.lower() in holiday_name_clean.lower():
            return 0.3, 1, holiday_name_clean

    for keyword in MODERATE_FESTIVALS:
        if keyword.lower() in holiday_name_clean.lower():
            return 0.5, 1, holiday_name_clean

    for keyword in MINOR_RELIGIOUS:
        if keyword.lower() in holiday_name_clean.lower():
            return 0.2, 1, holiday_name_clean

    return 0.25, 1, holiday_name_clean


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return "Smart Dustbin ML Backend Running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # --------------------------------------------------
        # Extract Inputs
        # --------------------------------------------------
        wet_level = float(data.get("wet_level", 0))
        dry_level = float(data.get("dry_level", 0))
        avg_fill_rate_last_3h = float(data.get("avg_fill_rate_last_3h", 0))
        previous_day_same_time_level = float(
            data.get("previous_day_same_time_level", 0)
        )
        weather_condition = str(data.get("weather_condition", "normal"))

        # --------------------------------------------------
        # Use IST Time (Cloud Driven)
        # --------------------------------------------------
        now = datetime.now(IST)

        current_date_str = now.strftime("%d-%m-%Y")
        current_time_str = now.strftime("%H:%M:%S")
        day_name_str = now.strftime("%A")

        # --------------------------------------------------
        # Holiday Engine
        # --------------------------------------------------
        holiday_factor, is_holiday, holiday_name = get_holiday_factor(
            now.date()
        )

        # --------------------------------------------------
        # Feature DataFrame
        # --------------------------------------------------
        input_df = pd.DataFrame([{
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "holiday_factor": float(holiday_factor),
            "is_holiday": int(is_holiday),
            "weather_condition": weather_condition,
            "wet_level": float(wet_level),
            "dry_level": float(dry_level),
            "avg_fill_rate_last_3h": float(avg_fill_rate_last_3h),
            "previous_day_same_time_level": float(previous_day_same_time_level)
        }])

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------
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

        pickup_required_immediately = bool(
            final_hours <= ALERT_LIMIT_HOURS
        )

        # --------------------------------------------------
        # RESPONSE (Everything preserved + date/time added)
        # --------------------------------------------------
        return jsonify({
            "selected_bin_for_pickup": str(selected_bin),
            "wet_predicted_rate": float(wet_rate),
            "dry_predicted_rate": float(dry_rate),
            "wet_hours_remaining": float(wet_hours),
            "dry_hours_remaining": float(dry_hours),
            "final_hours_remaining": float(final_hours),
            "next_pickup_datetime": pickup_time.strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_required_immediately": pickup_required_immediately,
            "is_holiday_today": bool(is_holiday),
            "holiday_name": str(holiday_name) if holiday_name else None,
            "holiday_factor_used": float(holiday_factor),

            # NEW CLOUD DRIVEN TIME
            "current_date": current_date_str,
            "current_time": current_time_str,
            "day_name": day_name_str
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# PRODUCTION ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)