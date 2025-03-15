from flask import Flask, render_template, jsonify, request
import numpy as np
import random
import json
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# Simple mock data
prediction_data = {
    "market": {
        "direction": "up",
        "magnitude": 0.85,
        "confidence": 0.78,
        "prices": {"BTC": 62500, "ETH": 3400, "LINK": 14.5, "BNB": 560, "SOL": 135}
    },
    "behavior": {
        "behavior": "positive",
        "confidence": 0.67,
        "market_sentiment": 0.32,
        "fear_greed_index": 65.0
    },
    "disease": {
        "r_value": 0.95,
        "new_cases_trend": -0.05,
        "peak_time_days": 0,
        "severity_index": 0.45,
        "containment_effectiveness": 0.78,
        "confidence": 0.82
    },
    "weather": {
        "forecast": [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "temperature": 20 + i + random.uniform(-3, 3),
                "precipitation_probability": min(1.0, 0.3 + i*0.05),
                "extreme_event_probability": 0.05 + i*0.01
            } for i in range(5)
        ]
    }
}

# Add advanced predictions
prediction_data["synthetic_reality"] = {
    "scenarios": [
        {
            "name": "Russia-China Trade Agreement",
            "impact": {"global_market": "positive", "usd_value": -0.03, "commodity_prices": 0.05},
            "probability": 0.65
        },
        {
            "name": "Fed Interest Rate Change",
            "impact": {"stock_market": "bearish", "housing_market": -0.08, "lending_rate": 0.01},
            "probability": 0.75
        }
    ]
}

prediction_data["neural_link"] = {
    "user_type": "trader",
    "decision_context": "Bitcoin investment",
    "subconscious_biases": {"risk_aversion": 0.6, "recency_bias": 0.7, "herd_mentality": 0.5},
    "recommendation": "buy",
    "confidence": 0.82
}

@app.route('/')
def index():
    print("Index route called")
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    print("Predictions API called")
    return jsonify(prediction_data)

@app.route('/api/history')
def get_history():
    prediction_type = request.args.get('type', 'market')
    print(f"History API called for {prediction_type}")
    # Mock history data - just return the same prediction 5 times with different timestamps
    history = []
    for i in range(5):
        entry = prediction_data.get(prediction_type, {}).copy()
        entry["timestamp"] = (datetime.now() - timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S")
        history.append(entry)
    return jsonify(history)

@app.route('/api/chart_data')
def get_chart_data():
    print("Chart data API called")
    # Mock chart data
    return jsonify({
        "x": [(datetime.now() - timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S") for i in range(5)],
        "y": [random.uniform(0.7, 0.9) for _ in range(5)]
    })

def update_mock_data():
    """Periodically update some values to simulate real-time changes"""
    while True:
        try:
            # Update market direction randomly
            if random.random() > 0.8:
                prediction_data["market"]["direction"] = random.choice(["up", "down"])
            
            # Update confidence values
            prediction_data["market"]["confidence"] = min(0.95, max(0.5, prediction_data["market"]["confidence"] + random.uniform(-0.05, 0.05)))
            prediction_data["behavior"]["confidence"] = min(0.95, max(0.5, prediction_data["behavior"]["confidence"] + random.uniform(-0.05, 0.05)))
            prediction_data["disease"]["confidence"] = min(0.95, max(0.5, prediction_data["disease"]["confidence"] + random.uniform(-0.05, 0.05)))
            
            # Update prices
            for asset in prediction_data["market"]["prices"]:
                change_pct = random.uniform(-0.02, 0.02)
                prediction_data["market"]["prices"][asset] *= (1 + change_pct)
                prediction_data["market"]["prices"][asset] = round(prediction_data["market"]["prices"][asset], 2)
            
            print(f"Updated mock data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(20)
        except Exception as e:
            print(f"Error updating mock data: {str(e)}")
            time.sleep(10)

if __name__ == '__main__':
    print("Starting simplified Quantum AI + On-Chain Data Prediction System Web Interface...")
    
    # Start background update thread
    update_thread = threading.Thread(target=update_mock_data)
    update_thread.daemon = True
    update_thread.start()
    
    # Run the app on port 8080
    app.run(debug=False, host='0.0.0.0', port=8080) 