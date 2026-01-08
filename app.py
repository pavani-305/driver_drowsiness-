from flask import Flask, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("knn_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

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
        elif result == 1:
            prediction = "DROWSY – Driver needs rest"
        else:
            prediction = "MICROSLEEP – Immediate action required"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Driver Fatigue Detection</title>
        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #f6f8fc;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}

            .container {{
                background: #ffffff;
                padding: 36px 32px;
                width: 440px;
                border-radius: 14px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            }}

            h2 {{
                text-align: center;
                color: #3a3a3a;
                margin-bottom: 28px;
            }}

            .form-group {{
                margin-bottom: 16px;
            }}

            label {{
                display: block;
                margin-bottom: 6px;
                font-size: 14px;
                color: #555;
            }}

            input {{
                width: 100%;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid #dcdcdc;
                font-size: 14px;
            }}

            button {{
                width: 100%;
                padding: 12px;
                margin-top: 10px;
                background-color: #b8c4f0;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
            }}

            .result {{
                margin-top: 20px;
                text-align: center;
                font-size: 16px;
                font-weight: 600;
                color: #333;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h2>Driver Fatigue Detection System</h2>

            <form method="POST">
                <input name="avg_EAR" placeholder="Average EAR" required>
                <input name="blink_rate" placeholder="Blink Rate" required>
                <input name="avg_blink_duration" placeholder="Avg Blink Duration" required>
                <input name="eye_closure_percentage" placeholder="Eye Closure %" required>
                <input name="long_eye_closure_count" placeholder="Long Eye Closure Count" required>
                <input name="head_nod_count" placeholder="Head Nod Count" required>
                <input name="head_pitch_variance" placeholder="Head Pitch Variance" required>
                <input name="perclos_30s" placeholder="PERCLOS (30s)" required>
                <input name="max_eye_closure_duration" placeholder="Max Eye Closure Duration" required>
                <input name="head_drop_events" placeholder="Head Drop Events" required>

                <button type="submit">Predict Driver State</button>
            </form>

            <div class="result">{prediction}</div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
