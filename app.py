from flask import Flask, request
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("knn_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    status_class = ""

    if request.method == "POST":
        values = [
            float(request.form["avg_EAR"]),
            float(request.form["blink_rate"]),
            float(request.form["avg_blink_duration"]),
            float(request.form["eye_closure_percentage"]),
            float(request.form["long_eye_closure_count"]),
            float(request.form["head_nod_count"]),
            float(request.form["head_pitch_variance"]),
            float(request.form["perclos_30s"]),
            float(request.form["max_eye_closure_duration"]),
            float(request.form["head_drop_events"])
        ]

        values = scaler.transform(np.array(values).reshape(1, -1))
        result = model.predict(values)[0]

        if result == 0:
            prediction = "ALERT – Driver is attentive"
            status_class = "safe"
        elif result == 1:
            prediction = "DROWSY – Driver needs rest"
            status_class = "warning"
        else:
            prediction = "MICROSLEEP – Immediate action required"
            status_class = "danger"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Driver Fatigue Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {{
            margin: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #eef2ff, #fdfbff);
            font-family: 'Segoe UI', sans-serif;
        }}

        .card {{
            width: 420px;
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(12px);
            padding: 36px;
            border-radius: 18px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        }}

        h2 {{
            text-align: center;
            margin-bottom: 28px;
            color: #2e2e2e;
        }}

        .field {{
            margin-bottom: 14px;
        }}

        input {{
            width: 100%;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid #d6d6d6;
            font-size: 14px;
            transition: all 0.2s ease;
        }}

        input:focus {{
            outline: none;
            border-color: #7b8cff;
            box-shadow: 0 0 0 3px rgba(123,140,255,0.2);
        }}

        button {{
            width: 100%;
            margin-top: 16px;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #7b8cff, #a2b2ff);
            color: #fff;
            cursor: pointer;
            transition: transform 0.15s ease;
        }}

        button:hover {{
            transform: translateY(-2px);
        }}

        .result {{
            margin-top: 22px;
            padding: 14px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 15px;
        }}

        .safe {{
            background: #e7f8ee;
            color: #1b7f45;
        }}

        .warning {{
            background: #fff5db;
            color: #9c6b00;
        }}

        .danger {{
            background: #ffe3e3;
            color: #b40000;
        }}

        footer {{
            margin-top: 18px;
            text-align: center;
            font-size: 12px;
            color: #777;
        }}
    </style>
</head>

<body>
    <div class="card">
        <h2>🚗 Driver Fatigue Detection</h2>

        <form method="POST">
            <div class="field"><input name="avg_EAR" placeholder="Average EAR" required></div>
            <div class="field"><input name="blink_rate" placeholder="Blink Rate (per min)" required></div>
            <div class="field"><input name="avg_blink_duration" placeholder="Avg Blink Duration (sec)" required></div>
            <div class="field"><input name="eye_closure_percentage" placeholder="Eye Closure %" required></div>
            <div class="field"><input name="long_eye_closure_count" placeholder="Long Eye Closure Count" required></div>
            <div class="field"><input name="head_nod_count" placeholder="Head Nod Count" required></div>
            <div class="field"><input name="head_pitch_variance" placeholder="Head Pitch Variance" required></div>
            <div class="field"><input name="perclos_30s" placeholder="PERCLOS (30s)" required></div>
            <div class="field"><input name="max_eye_closure_duration" placeholder="Max Eye Closure Duration" required></div>
            <div class="field"><input name="head_drop_events" placeholder="Head Drop Events" required></div>

            <button type="submit">Predict Driver State</button>
        </form>

        {f'<div class="result {status_class}">{prediction}</div>' if prediction else ''}

        <footer>ML-based Driver Monitoring System</footer>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
