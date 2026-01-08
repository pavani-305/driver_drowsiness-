from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

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

            input:focus {{
                outline: none;
                border-color: #b8c4f0;
                box-shadow: 0 0 0 2px rgba(184,196,240,0.3);
            }}

            button {{
                width: 100%;
                padding: 12px;
                margin-top: 10px;
                background-color: #b8c4f0;
                color: #2f2f2f;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
            }}

            button:hover {{
                background-color: #a7b5ec;
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

                <div class="form-group">
                    <label>Average Eye Aspect Ratio (EAR)</label>
                    <input name="avg_EAR" required>
                </div>

                <div class="form-group">
                    <label>Blink Rate (per minute)</label>
                    <input name="blink_rate" required>
                </div>

                <div class="form-group">
                    <label>Average Blink Duration (seconds)</label>
                    <input name="avg_blink_duration" required>
                </div>

                <div class="form-group">
                    <label>Eye Closure Percentage (%)</label>
                    <input name="eye_closure_percentage" required>
                </div>

                <div class="form-group">
                    <label>Long Eye Closure Count</label>
                    <input name="long_eye_closure_count" required>
                </div>

                <div class="form-group">
                    <label>Head Nod Count</label>
                    <input name="head_nod_count" required>
                </div>

                <div class="form-group">
                    <label>Head Pitch Variance</label>
                    <input name="head_pitch_variance" required>
                </div>

                <div class="form-group">
                    <label>PERCLOS (last 30 seconds)</label>
                    <input name="perclos_30s" required>
                </div>

                <div class="form-group">
                    <label>Maximum Eye Closure Duration (seconds)</label>
                    <input name="max_eye_closure_duration" required>
                </div>

                <div class="form-group">
                    <label>Head Drop Events</label>
                    <input name="head_drop_events" required>
                </div>

                <button type="submit">Predict Driver State</button>
            </form>

            <div class="result">{prediction}</div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
