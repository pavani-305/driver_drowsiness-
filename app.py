from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load model and scaler trained with ALL features
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

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

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

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Driver Fatigue Detection System</title>

    <style>
        * {
            box-sizing: border-box;
        }

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

        /* LEFT PANEL */
        .left-panel {
            padding: 50px 40px;
            background: linear-gradient(180deg, rgba(255,255,255,0.15), rgba(255,255,255,0.02));
        }

        .left-panel h1 {
            font-size: 36px;
            margin-bottom: 20px;
            line-height: 1.3;
        }

        .left-panel p {
            font-size: 15px;
            color: #e0e0e0;
            line-height: 1.7;
            margin-bottom: 30px;
        }

        .info-box {
            background: rgba(0, 0, 0, 0.25);
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 18px;
            font-size: 14px;
        }

        .info-box span {
            font-weight: bold;
            color: #8fd3f4;
        }

        /* RIGHT PANEL */
        .right-panel {
            padding: 40px 36px;
            background: rgba(0, 0, 0, 0.35);
        }

        .right-panel h2 {
            text-align: center;
            margin-bottom: 26px;
            font-size: 26px;
        }

        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px 18px;
        }

        label {
            font-size: 13px;
            margin-bottom: 4px;
            display: block;
            color: #cfdfff;
        }

        input {
            width: 100%;
            padding: 10px 12px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            background: rgba(255,255,255,0.85);
        }

        input:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(143, 211, 244, 0.6);
        }

        .full-width {
            grid-column: span 2;
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
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(143, 211, 244, 0.5);
        }

        .result-box {
            margin-top: 22px;
            padding: 16px;
            border-radius: 14px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(6px);
        }

        .footer-note {
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #cccccc;
        }

        @media (max-width: 1024px) {
            .main-card {
                grid-template-columns: 1fr;
                width: 92%;
            }
        }
    </style>
</head>

<body>

<div class="main-card">

    <!-- LEFT INFORMATION PANEL -->
    <div class="left-panel">
        <h1>AI-Based<br>Driver Fatigue<br>Detection</h1>

        <p>
            This system uses machine learning and behavioral analysis to
            detect driver fatigue in real time by analyzing eye movement,
            head posture, and facial activity patterns.
        </p>

        <div class="info-box">
            <span>🧠 Model:</span> K-Nearest Neighbors (KNN)
        </div>

        <div class="info-box">
            <span>📊 Features:</span> Eye, Head & Mouth Behaviour
        </div>

        <div class="info-box">
            <span>🎯 Output:</span> Alert / Drowsy / Microsleep
        </div>

        <div class="info-box">
            <span>🚗 Use Case:</span> Accident Prevention & Road Safety
        </div>
    </div>

    <!-- RIGHT FORM PANEL -->
    <div class="right-panel">
        <h2>Driver Behaviour Input</h2>

        <form method="POST">

            <div>
                <label>Average EAR</label>
                <input name="avg_EAR" required>
            </div>

            <div>
                <label>Blink Rate</label>
                <input name="blink_rate" required>
            </div>

            <div>
                <label>Avg Blink Duration</label>
                <input name="avg_blink_duration" required>
            </div>

            <div>
                <label>Eye Closure %</label>
                <input name="eye_closure_percentage" required>
            </div>

            <div>
                <label>Long Eye Closures</label>
                <input name="long_eye_closure_count" required>
            </div>

            <div>
                <label>Head Nod Count</label>
                <input name="head_nod_count" required>
            </div>

            <div>
                <label>Head Pitch Variance</label>
                <input name="head_pitch_variance" required>
            </div>

            <div>
                <label>PERCLOS (30s)</label>
                <input name="perclos_30s" required>
            </div>

            <div>
                <label>Max Eye Closure Duration</label>
                <input name="max_eye_closure_duration" required>
            </div>

            <div>
                <label>Head Drop Events</label>
                <input name="head_drop_events" required>
            </div>

            <div>
                <label>Yawn Count</label>
                <input name="yawn_count" required>
            </div>

            <div>
                <label>Mouth Open Duration</label>
                <input name="mouth_open_duration" required>
            </div>

            <div>
                <label>Mouth Open Ratio</label>
                <input name="mouth_open_ratio" required>
            </div>

            <div>
                <label>Hand Near Mouth (0/1)</label>
                <input name="hand_near_mouth_flag" required>
            </div>

            <button type="submit">Analyze Driver State</button>
        </form>

        <div class="result-box">
            {{ prediction }}
        </div>

        <div class="footer-note">
            AI-Driven Road Safety System • Academic Project
        </div>
    </div>

</div>

</body>
</html>

"""

if __name__ == "__main__":
    app.run(debug=True)
