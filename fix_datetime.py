from flask import Flask, render_template, redirect, url_for, request, session, jsonify, flash
from functools import wraps
import os
from datetime import datetime, timedelta
import json
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)  # Enable CORS for API endpoints

# Login check decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple authentication for demo
        if username == "demo" and password == "password123":
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('dashboard.html', username=session.get('username', 'User'), current_time=current_time)

@app.route('/portfolio')
@login_required
def portfolio():
    portfolio_data = {
        'cryptocurrencies': [
            {'name': 'Bitcoin', 'symbol': 'BTC', 'amount': 0.5, 'value_usd': 29000.00, 'change_24h': 2.5},
            {'name': 'Ethereum', 'symbol': 'ETH', 'amount': 5.0, 'value_usd': 3000.00, 'change_24h': 1.8},
            {'name': 'Solana', 'symbol': 'SOL', 'amount': 25.0, 'value_usd': 100.00, 'change_24h': 5.2}
        ],
        'stocks': [
            {'name': 'Apple Inc.', 'symbol': 'AAPL', 'shares': 10, 'value_usd': 175.00, 'change_24h': 0.3},
            {'name': 'Tesla Inc.', 'symbol': 'TSLA', 'shares': 5, 'value_usd': 250.00, 'change_24h': -1.2},
            {'name': 'Microsoft', 'symbol': 'MSFT', 'shares': 8, 'value_usd': 320.00, 'change_24h': 0.9}
        ]
    }
    return render_template('portfolio.html', portfolio=portfolio_data)

@app.route('/sports')
@login_required
def sports():
    matches = [
        {
            'league': 'English Premier League',
            'teams': 'Arsenal vs Manchester United',
            'match_time': '2023-11-10 15:00',
            'odds': {'Home': 2.1, 'Draw': 3.5, 'Away': 2.8},
            'prediction': 'Home Win',
            'confidence': 0.75
        },
        {
            'league': 'NBA',
            'teams': 'LA Lakers vs Chicago Bulls',
            'match_time': '2023-11-11 19:30',
            'odds': {'Home': 1.8, 'Away': 2.1},
            'prediction': 'Home Win',
            'confidence': 0.65
        },
        {
            'league': 'NFL',
            'teams': 'Kansas City Chiefs vs Buffalo Bills',
            'match_time': '2023-11-12 20:00',
            'odds': {'Home': 1.9, 'Away': 1.95},
            'prediction': 'Away Win',
            'confidence': 0.55
        }
    ]
    return render_template('sports.html', matches=matches)

@app.route('/weather')
@login_required
def weather():
    forecast = [
        {'date': '2023-11-09', 'description': 'Sunny', 'temperature': 22.5, 'precipitation_probability': 0.05, 'wind_speed': 8.2, 'humidity': 45},
        {'date': '2023-11-10', 'description': 'Partly Cloudy', 'temperature': 20.1, 'precipitation_probability': 0.15, 'wind_speed': 10.5, 'humidity': 50},
        {'date': '2023-11-11', 'description': 'Rain', 'temperature': 18.3, 'precipitation_probability': 0.70, 'wind_speed': 15.0, 'humidity': 80},
        {'date': '2023-11-12', 'description': 'Heavy Rain', 'temperature': 17.5, 'precipitation_probability': 0.85, 'wind_speed': 18.3, 'humidity': 85},
        {'date': '2023-11-13', 'description': 'Partly Cloudy', 'temperature': 19.8, 'precipitation_probability': 0.30, 'wind_speed': 12.1, 'humidity': 60}
    ]
    return render_template('weather.html', forecast=forecast)

@app.route('/health')
@login_required
def health():
    diseases = [
        {
            'name': 'COVID-19',
            'r_value': 0.95,
            'trend': 'Decreasing',
            'severity': 'Moderate',
            'containment_level': 'High',
            'regions_affected': ['North America', 'Europe', 'Asia']
        },
        {
            'name': 'Influenza',
            'r_value': 1.2,
            'trend': 'Increasing',
            'severity': 'Low',
            'containment_level': 'Moderate',
            'regions_affected': ['North America', 'Europe']
        },
        {
            'name': 'Dengue',
            'r_value': 1.4,
            'trend': 'Stable',
            'severity': 'High',
            'containment_level': 'Low',
            'regions_affected': ['South America', 'Southeast Asia', 'Africa']
        }
    ]
    return render_template('health.html', diseases=diseases)

@app.route('/conflicts')
@login_required
def conflicts():
    global_risk_index = 68
    conflicts = [
        {
            'region': 'Eastern Europe',
            'risk_level': 'High',
            'risk_percentage': 85,
            'trend': 'Increasing',
            'key_factors': ['Military Buildup', 'Diplomatic Tensions', 'Economic Sanctions'],
            'prediction': 'Continued escalation with limited direct confrontation'
        },
        {
            'region': 'Middle East',
            'risk_level': 'Medium',
            'risk_percentage': 60,
            'trend': 'Stable',
            'key_factors': ['Resource Disputes', 'Historical Tensions', 'Political Instability'],
            'prediction': 'Localized conflicts with potential for regional impact'
        },
        {
            'region': 'South China Sea',
            'risk_level': 'Medium',
            'risk_percentage': 55,
            'trend': 'Increasing',
            'key_factors': ['Territorial Disputes', 'Trade Route Control', 'Military Presence'],
            'prediction': 'Increased naval activity with diplomatic resolutions likely'
        },
        {
            'region': 'North Africa',
            'risk_level': 'Low',
            'risk_percentage': 30,
            'trend': 'Decreasing',
            'key_factors': ['Political Transitions', 'Economic Development', 'International Support'],
            'prediction': 'Improving stability with localized exceptions'
        }
    ]
    return render_template('conflicts.html', global_risk_index=global_risk_index, conflicts=conflicts)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/api/predictions')
@login_required
def get_predictions():
    # Current predictions data
    predictions = {
        'market': {
            'direction': 'up',
            'magnitude': 0.78,
            'confidence': 0.82,
            'timestamp': datetime.now().isoformat(),
            'prices': {
                'AAPL': {'price': 175.23, 'change_pct': 0.5},
                'MSFT': {'price': 320.45, 'change_pct': 0.8},
                'GOOG': {'price': 132.10, 'change_pct': -0.3},
                'AMZN': {'price': 145.67, 'change_pct': 1.2},
                'TSLA': {'price': 248.90, 'change_pct': -1.2},
                'META': {'price': 325.65, 'change_pct': 2.1}
            }
        },
        'behavior': {
            'behavior': 'positive',
            'confidence': 0.75,
            'fear_greed_index': 65,
            'news': [
                {
                    'title': 'Markets reach new highs amid positive economic data',
                    'url': '#',
                    'publishedAt': datetime.now().isoformat()
                },
                {
                    'title': 'Fed signals potential rate cut in upcoming meeting',
                    'url': '#',
                    'publishedAt': datetime.now().isoformat()
                },
                {
                    'title': 'Tech stocks rally on strong earnings reports',
                    'url': '#',
                    'publishedAt': datetime.now().isoformat()
                }
            ]
        }
    }
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 