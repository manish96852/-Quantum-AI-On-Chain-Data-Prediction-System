import asyncio
from quantum_predictor.prediction_engine import PredictionEngine

async def main():
    # Initialize prediction engine with your Web3 provider and Etherscan API key
    engine = PredictionEngine(
        web3_provider_url="YOUR_WEB3_PROVIDER_URL",  # e.g., Infura endpoint
        etherscan_api_key="YOUR_ETHERSCAN_API_KEY"
    )
    
    # Example: Make market prediction
    market_prediction = await engine.predict("market")
    print("\nMarket Prediction:")
    print(f"Direction: {market_prediction['prediction']['direction']}")
    print(f"Magnitude: {market_prediction['prediction']['magnitude']:.2f}")
    print(f"Confidence: {market_prediction['confidence']:.2f}")
    
    # Example: Make sports prediction
    sports_prediction = await engine.predict("sports")
    print("\nSports Prediction:")
    print(f"Win Probability: {sports_prediction['prediction']['win_probability']:.2%}")
    print(f"Confidence: {sports_prediction['confidence']:.2f}")
    
    # Example: Make weather prediction
    weather_prediction = await engine.predict("weather")
    print("\nWeather Prediction:")
    print(f"Precipitation Probability: {weather_prediction['prediction']['precipitation_probability']:.2%}")
    print(f"Confidence: {weather_prediction['confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 