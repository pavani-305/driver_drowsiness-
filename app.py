from flask import Flask, request, render_template_string
import numpy as np
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
model = joblib.load("knn_fatigue_model_all_features.pkl")
scaler = joblib.load("scaler_all_features.pkl")

# --------------------------------------------------
# EXACT feature order used during training
# --------------------------------------------------
FEATURE_NAMES = [
    "avg_EAR",
    "blink_rate",
    "avg_blink_duration",
    "eye_closure_percentage",
    "long_eye_closure_count",
    "head_nod_count",
    "head_pitch_variance",
    "perclos_30s",
    "max_eye_closure_duration",
    "head_drop_events",
    "yawn_count",
    "mouth_open_duration",
    "mouth_open_ratio",
    "hand_near_mouth_flag"
]

# --------------------------------------------------
# MAIN ROUTE
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = "Enter values and click Analyze"

    if request.method == "POST":
        try:
            # Collect inputs in SAME ORDER as training
            values = [float(request.form[f]) for f in FEATURE_NAMES]

            # Scale inputs
            values_scaled = scaler.transform(
                np.array(values).reshape(1, -1)
            )

            # Predict
            result = model.predict(values_scaled)[0]

            if result == 0:
                prediction = "🟢 ALERT – Driver is Attentive"
            elif result == 1:
                prediction = "🟠 DROWSY – Driver Needs Rest"
            else:
                prediction = "🔴 MICROSLEEP – Immediate Action Required"

        except Exception as e:
            prediction = f"Input Error: {e}"

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Driver Fatigue Detection</title>

    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            display: flex;
            justify-content: center;
            align-items: center;
            color: #ffffff;
        }

        .main-card {
            width: 1100px;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            border-radius: 22px;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
            display: grid;
            grid-template-columns: 1fr 1.4fr;
            overflow: hidden;
        }

        .left-panel {
            padding: 50px 40px;
            background: linear-gradient(180deg, rgba(255,255,255,0.15), rgba(255,255,255,0.02));
        }

        .left-panel h1 {
            font-size: 36px;
            margin-bottom: 20px;
        }

        .left-panel p {
            font-size: 15px;
            color: #e0e0e0;
            line-height: 1.7;
        }

        .info-box {
            background: rgba(0, 0, 0, 0.25);
            padding: 18px;
            border-radius: 14px;
            margin-top: 14px;
            font-size: 14px;
        }

        .right-panel {
            padding: 40px 36px;
            background: rgba(0, 0, 0, 0.35);
        }

        h2 {
            text-align: center;
            margin-bottom: 26px;
        }

        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px 18px;
        }

        label {
            font-size: 13px;
            color: #cfdfff;
        }

        input {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: none;
        }

        button {
            grid-column: span 2;
            margin-top: 14px;
            padding: 14px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 14px;
            border: none;
            cursor: pointer;
            background: linear-gradient(135deg, #8fd3f4, #84fab0);
            color: #0f2027;
        }

        .result-box {
            margin-top: 22px;
            padding: 16px;
            border-radius: 14px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.15);
        }

        @media (max-width: 1024px) {
            .main-card {
                grid-template-columns: 1fr;
                width: 95%;
            }
        }
    </style>
</head>

<body>

<div class="main-card">

    <div class="left-panel">
        <h1>AI Driver<br>Fatigue Detection</h1>
        <p>
            Machine learning based system that detects driver fatigue
            using eye, head and mouth behaviour patterns.
        </p>

        <div class="info-box">🧠 Model: KNN</div>
        <div class="info-box">📊 Inputs: 14 Behaviour Features</div>
        <div class="info-box">🚗 Use: Road Safety</div>
    </div>

    <div class="right-panel">
        <h2>Driver Inputs</h2>

        <form method="POST">
            {% for f in features %}
            <div>
                <label>{{ f }}</label>
                <input name="{{ f }}" required>
            </div>
            {% endfor %}

            <button type="submit">Analyze Driver State</button>
        </form>

        <div class="result-box">
            {{ prediction }}
        </div>
    </div>

</div>

</body>
</html>
""", prediction=prediction, features=FEATURE_NAMES)

# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
