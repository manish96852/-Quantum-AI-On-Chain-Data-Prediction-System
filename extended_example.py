import numpy as np
import random
import time
from datetime import datetime, timedelta

class SimpleQuantumSimulator:
    """
    A simplified version of the quantum circuit without external dependencies
    """
    def __init__(self, n_qubits=4):
        """
        Initialize the simulator
        n_qubits: Number of qubits to simulate
        """
        self.n_qubits = n_qubits
        
    def apply_quantum_rotation(self, inputs):
        """
        Simulate a quantum rotation operation
        In real implementation this would use actual quantum gates
        """
        # Normalize inputs to [-1, 1] range
        normalized_inputs = np.clip(inputs, -np.pi, np.pi) / np.pi
        
        # Simulate quantum measurement outcomes
        # In a real quantum circuit, this would be the result of actual measurements
        simulated_measurements = []
        for i in range(self.n_qubits):
            # Apply a simplified rotation and measurement model
            angle = normalized_inputs[i % len(normalized_inputs)]
            # Approximate quantum probability using classical formula
            prob = (np.cos(angle) ** 2)
            # Convert to [-1, 1] range for measurement outcome
            measurement = 2 * prob - 1
            simulated_measurements.append(measurement)
            
        return np.array(simulated_measurements)
    
    def predict(self, data):
        """
        Make prediction based on input data
        data: Input data for prediction
        """
        # Ensure we have enough data (pad if necessary)
        padded_data = np.zeros(self.n_qubits)
        padded_data[:min(len(data), self.n_qubits)] = data[:min(len(data), self.n_qubits)]
        
        # Get simulated quantum measurements
        measurements = self.apply_quantum_rotation(padded_data)
        
        # Calculate prediction probability
        prediction_value = np.mean(measurements)
        
        # The confidence based on the spread of measurements
        confidence = 1.0 - np.std(measurements)
        
        return {
            "prediction": prediction_value,
            "direction": "up" if prediction_value > 0 else "down",
            "magnitude": abs(prediction_value),
            "confidence": confidence
        }


class OnChainDataSimulator:
    """
    Simulates blockchain data without actual blockchain dependencies
    """
    def __init__(self):
        """Initialize simulator"""
        # Simulated price data
        self.assets = {
            "BTC": 63400 + random.uniform(-1000, 1000),
            "ETH": 3500 + random.uniform(-200, 200),
            "LINK": 15 + random.uniform(-1, 1),
            "BNB": 570 + random.uniform(-20, 20),
            "SOL": 140 + random.uniform(-10, 10)
        }
        
        # Simulated DeFi metrics
        self.defi_metrics = {
            "tvl": 48.7 + random.uniform(-2, 2),  # Total Value Locked in billions
            "daily_volume": 3.2 + random.uniform(-0.5, 0.5),  # Daily volume in billions
            "unique_addresses": 123400 + random.randint(-5000, 5000)
        }
        
        # Simulated sports betting odds
        self.sports_odds = {
            "team_a_win": 1.8 + random.uniform(-0.2, 0.2),
            "team_b_win": 2.1 + random.uniform(-0.2, 0.2),
            "draw": 3.5 + random.uniform(-0.3, 0.3)
        }
    
    def get_price_data(self):
        """Get simulated price data"""
        # Add small random fluctuations to prices to simulate real-time changes
        for asset in self.assets:
            change_pct = random.uniform(-0.01, 0.01)  # -1% to +1%
            self.assets[asset] *= (1 + change_pct)
            
        return self.assets
    
    def get_defi_data(self):
        """Get simulated DeFi metrics"""
        # Add small fluctuations
        self.defi_metrics["tvl"] *= (1 + random.uniform(-0.005, 0.005))
        self.defi_metrics["daily_volume"] *= (1 + random.uniform(-0.01, 0.01))
        self.defi_metrics["unique_addresses"] += random.randint(-100, 100)
        
        return self.defi_metrics
    
    def get_sports_data(self):
        """Get simulated sports betting data"""
        # Add small fluctuations to odds
        for key in self.sports_odds:
            self.sports_odds[key] += random.uniform(-0.05, 0.05)
            
        return self.sports_odds
    
    def prepare_features(self):
        """Prepare features for prediction"""
        # Get the data
        prices = self.get_price_data()
        defi = self.get_defi_data()
        sports = self.get_sports_data()
        
        # Create feature vector (normalized values)
        features = []
        
        # Add price changes (normalized)
        for asset, price in prices.items():
            # Simulate 24h change
            change_24h = random.uniform(-0.1, 0.1)
            normalized_change = change_24h * 10  # Scale to roughly -1 to 1
            features.append(normalized_change)
        
        # Add DeFi metrics (normalized)
        features.append(defi["tvl"] / 50)  # Normalize by dividing by expected max
        features.append(defi["daily_volume"] / 5)
        features.append(defi["unique_addresses"] / 200000)
        
        # Add sports odds (inverse of odds, normalized)
        for key, odds in sports.items():
            features.append((1/odds) - 0.3)  # Center around 0
            
        return features


class BehavioralAnalysisSimulator:
    """
    Simulates behavioral analysis without NLP dependencies
    """
    def __init__(self):
        """Initialize simulator"""
        # Simulated sentiment data
        self.sentiment_data = {
            "market_sentiment": random.uniform(-1, 1),  # -1 (bearish) to 1 (bullish)
            "social_media_volume": random.uniform(0.3, 1.0),  # 0 (low) to 1 (high)
            "news_sentiment": random.uniform(-1, 1),  # -1 (negative) to 1 (positive)
            "expert_consensus": random.uniform(-1, 1),  # -1 (bearish) to 1 (bullish)
            "fear_greed_index": random.uniform(20, 80)  # 0 (extreme fear) to 100 (extreme greed)
        }
        
    def analyze_market_sentiment(self):
        """Analyze market sentiment"""
        # Add small random fluctuations to sentiment
        for key in self.sentiment_data:
            if key == "fear_greed_index":
                self.sentiment_data[key] += random.uniform(-5, 5)
                self.sentiment_data[key] = max(0, min(100, self.sentiment_data[key]))
            else:
                self.sentiment_data[key] += random.uniform(-0.1, 0.1)
                self.sentiment_data[key] = max(-1, min(1, self.sentiment_data[key]))
                
        return self.sentiment_data
    
    def predict_human_behavior(self):
        """Predict human behavior based on sentiment"""
        sentiment = self.analyze_market_sentiment()
        
        # Average the sentiment metrics
        avg_sentiment = (
            sentiment["market_sentiment"] + 
            sentiment["news_sentiment"] + 
            sentiment["expert_consensus"] +
            (sentiment["fear_greed_index"] / 50 - 1)
        ) / 4
        
        # Sentiment volatility (simulate)
        volatility = random.uniform(0.1, 0.5)
        
        # Calculate behavior prediction
        behavior = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
        confidence = 0.5 + abs(avg_sentiment)/2
        confidence *= (1 - volatility/2)  # Reduce confidence if high volatility
        
        return {
            "behavior": behavior,
            "confidence": min(0.95, confidence),
            "sentiment": sentiment,
            "volatility": volatility
        }


class WeatherDiseasePredictionSimulator:
    """
    Simulates weather and disease prediction without specialized dependencies
    """
    def __init__(self):
        """Initialize simulator"""
        self.current_date = datetime.now()
        
        # Simulated current weather data
        self.current_weather = {
            "temperature": 22 + random.uniform(-5, 5),
            "humidity": 65 + random.uniform(-10, 10),
            "pressure": 1012 + random.uniform(-10, 10),
            "wind_speed": 15 + random.uniform(-5, 5),
            "cloud_cover": random.uniform(0, 100)
        }
        
        # Simulated disease metrics
        self.disease_metrics = {
            "r_value": 0.9 + random.uniform(-0.3, 0.3),  # R0 reproduction number
            "cases_per_100k": 12 + random.uniform(-5, 5),
            "hospital_capacity": 70 + random.uniform(-10, 10),  # percentage
            "vaccination_rate": 65 + random.uniform(-10, 10)    # percentage
        }
    
    def predict_weather(self, days_ahead=5):
        """Predict weather for given days ahead"""
        predictions = []
        
        # Base values
        base_temp = self.current_weather["temperature"]
        base_humidity = self.current_weather["humidity"]
        base_precip_prob = 0.3
        
        # For each day, generate a prediction
        for i in range(days_ahead):
            date = (self.current_date + timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Add some randomness and trend
            temp = base_temp + i*0.5 + random.uniform(-3, 3)
            humidity = base_humidity + random.uniform(-5, 5)
            precip_prob = min(1.0, max(0.0, base_precip_prob + 0.05*i + random.uniform(-0.1, 0.1)))
            
            # Calculate extreme weather probability
            extreme_prob = max(0, min(1, 0.05 + i*0.01 + random.uniform(-0.02, 0.02)))
            if temp > 30 or temp < 0:
                extreme_prob += 0.1  # Increase probability for extreme temperatures
                
            predictions.append({
                "date": date,
                "temperature": temp,
                "humidity": humidity,
                "precipitation_probability": precip_prob,
                "extreme_event_probability": extreme_prob
            })
            
        return predictions
    
    def predict_disease_spread(self):
        """Predict disease spread"""
        # Add small fluctuations to disease metrics
        for key in self.disease_metrics:
            if key == "r_value":
                self.disease_metrics[key] += random.uniform(-0.05, 0.05)
                self.disease_metrics[key] = max(0, min(3, self.disease_metrics[key]))
            elif key in ["hospital_capacity", "vaccination_rate"]:
                self.disease_metrics[key] += random.uniform(-2, 2)
                self.disease_metrics[key] = max(0, min(100, self.disease_metrics[key]))
            else:
                self.disease_metrics[key] += random.uniform(-1, 1)
                self.disease_metrics[key] = max(0, self.disease_metrics[key])
        
        # Calculate trend in new cases
        r_value = self.disease_metrics["r_value"]
        new_cases_trend = r_value - 1  # Above 1 means growth, below 1 means decline
        
        # Predict days until peak (if r > 1, otherwise past peak)
        if r_value > 1:
            peak_time_days = int(20 + 10 * (1/(r_value-0.9)) + random.uniform(-5, 5))
            peak_time_days = max(0, peak_time_days)
        else:
            # Already declining
            peak_time_days = 0
            
        # Severity index (0-1)
        severity_index = (
            (self.disease_metrics["r_value"] / 3) * 0.4 +
            (self.disease_metrics["cases_per_100k"] / 50) * 0.3 +
            (1 - self.disease_metrics["hospital_capacity"] / 100) * 0.3
        )
        severity_index = max(0, min(1, severity_index))
        
        # Containment effectiveness (0-1)
        containment_effectiveness = (
            (self.disease_metrics["vaccination_rate"] / 100) * 0.5 +
            (1 - severity_index) * 0.3 +
            (1 - r_value/3) * 0.2
        )
        containment_effectiveness = max(0, min(1, containment_effectiveness))
        
        # Calculate confidence level
        confidence = 0.7 + random.uniform(-0.1, 0.2)
        confidence = min(0.95, max(0.5, confidence))
        
        return {
            "r_value": self.disease_metrics["r_value"],
            "new_cases_trend": new_cases_trend,
            "peak_time_days": peak_time_days,
            "severity_index": severity_index,
            "containment_effectiveness": containment_effectiveness,
            "confidence": confidence
        }


def run_simulation():
    """Run a simulation of the entire system"""
    print("Starting Quantum AI + On-Chain Data Prediction System Simulation...")
    print("Initializing modules...")
    
    # Initialize components
    quantum_sim = SimpleQuantumSimulator(n_qubits=8)
    blockchain_sim = OnChainDataSimulator()
    behavior_sim = BehavioralAnalysisSimulator()
    weather_disease_sim = WeatherDiseasePredictionSimulator()
    
    print("Loading on-chain data...")
    blockchain_features = blockchain_sim.prepare_features()
    
    print("Running quantum prediction algorithm...")
    market_prediction = quantum_sim.predict(blockchain_features)
    
    print("Analyzing behavioral patterns...")
    behavior_prediction = behavior_sim.predict_human_behavior()
    
    print("Processing epidemiological models...")
    epidemic_prediction = weather_disease_sim.predict_disease_spread()
    
    print("Running climate prediction models...")
    weather_predictions = weather_disease_sim.predict_weather(days_ahead=1)
    
    # Print summary results
    print("\n----------------------------------------")
    print("SIMULATION RESULTS SUMMARY")
    print("----------------------------------------")
    
    # Market prediction
    print("\n🔹 STOCK MARKET PREDICTION:")
    print(f"  Market Direction: {market_prediction['direction']}")
    print(f"  Magnitude: {market_prediction['magnitude']:.2f}")
    print(f"  Confidence: {market_prediction['confidence']:.2f}")
    
    # Show price data
    print("\n  Current Blockchain Price Data:")
    for asset, price in blockchain_sim.get_price_data().items():
        print(f"  - {asset}: ${price:.2f}")
    
    # Human behavior prediction
    print("\n🔹 HUMAN BEHAVIOR PREDICTION:")
    print(f"  Behavior Prediction: {behavior_prediction['behavior']}")
    print(f"  Confidence: {behavior_prediction['confidence']:.2f}")
    print(f"  Market Sentiment: {behavior_prediction['sentiment']['market_sentiment']:.2f}")
    print(f"  Fear & Greed Index: {behavior_prediction['sentiment']['fear_greed_index']:.1f}")
    
    # Disease outbreak prediction
    print("\n🔹 DISEASE OUTBREAK PREDICTION:")
    print(f"  R Value: {epidemic_prediction['r_value']:.2f}")
    print(f"  New Cases Trend: {epidemic_prediction['new_cases_trend']:.2f}")
    print(f"  Days Until Peak: {epidemic_prediction['peak_time_days']}")
    print(f"  Severity Index: {epidemic_prediction['severity_index']:.2f}")
    print(f"  Containment Effectiveness: {epidemic_prediction['containment_effectiveness']:.2f}")
    print(f"  Confidence: {epidemic_prediction['confidence']:.2f}")
    
    # Weather forecast
    print("\n🔹 WEATHER FORECAST:")
    for day in weather_predictions:
        print(f"  Temperature: {day['temperature']:.1f}°C")
        print(f"  Rain Chance: {day['precipitation_probability']*100:.1f}%")
        print(f"  Extreme Event Probability: {day['extreme_event_probability']*100:.2f}%")
    
    print("\n----------------------------------------")
    print("NOTE: This is a simplified demo without actual quantum computing or blockchain connections.")
    print("All predictions are simulated and not to be used for actual decisions.")
    print("----------------------------------------")
    
if __name__ == "__main__":
    run_simulation() 