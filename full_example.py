import asyncio
from quantum_predictor.prediction_engine import PredictionEngine
from quantum_predictor.psychological_analysis import BehavioralAnalyzer
from quantum_predictor.disease_weather_engine import EpidemicWeatherEngine
from quantum_predictor.enhanced_blockchain import SmartContractIntegration

async def run_comprehensive_predictions():
    print("====== QUANTUM AI + BLOCKCHAIN PREDICTION SYSTEM ======")
    
    # Initialize components
    # In a real implementation, these would use actual API keys and endpoints
    web3_provider_url = "YOUR_WEB3_PROVIDER_URL"  # e.g., Infura endpoint
    etherscan_api_key = "YOUR_ETHERSCAN_API_KEY"
    
    # Initialize main prediction engine
    print("\nInitializing Quantum AI Prediction Engine...")
    prediction_engine = PredictionEngine(web3_provider_url, etherscan_api_key)
    
    # Initialize behavioral analyzer
    print("Initializing AI-Driven Psychological & Behavioral Analysis Engine...")
    behavioral_analyzer = BehavioralAnalyzer()
    
    # Initialize epidemic and weather prediction engine
    print("Initializing Disease Outbreak & Weather Prediction Engine...")
    epidemic_weather_engine = EpidemicWeatherEngine()
    
    # Initialize blockchain smart contract integration
    print("Initializing Enhanced Blockchain Integration...")
    smart_contract = SmartContractIntegration(web3_provider_url)
    
    # Demo 1: Stock Market Prediction
    print("\n\n====== DEMO 1: STOCK MARKET PREDICTION ======")
    market_prediction = await prediction_engine.predict("market")
    print(f"Market Direction: {market_prediction['prediction']['direction']}")
    print(f"Magnitude: {market_prediction['prediction']['magnitude']:.2f}")
    print(f"Confidence: {market_prediction['confidence']:.2f}")
    
    # Get blockchain factors that influenced the prediction
    print("\nBlockchain Factors:")
    for asset, price in market_prediction['blockchain_factors']['prices'].items():
        print(f"  - {asset}: ${price:.2f}")
    
    # Demo 2: Sports Prediction
    print("\n\n====== DEMO 2: SPORTS BETTING PREDICTION ======")
    # Simulate fetching decentralized sports data
    defi_feeds = await smart_contract.fetch_decentralized_data_feeds()
    sports_feeds = defi_feeds.get("sports", {})
    
    print("Available Sports Data:")
    for sport, data in sports_feeds.items():
        print(f"  - {sport}: Odds: {data['odds']}")
    
    # Make sports prediction
    sports_prediction = await prediction_engine.predict("sports")
    print(f"\nWin Probability: {sports_prediction['prediction']['win_probability']:.2%}")
    print(f"Confidence: {sports_prediction['confidence']:.2f}")
    
    # Demo 3: Political Conflict Prediction
    print("\n\n====== DEMO 3: GLOBAL CONFLICT PREDICTION ======")
    # Choose a region for conflict analysis
    region = "MiddleEast"
    conflict_prediction = await behavioral_analyzer.analyze_political_conflict(region)
    
    print(f"Region: {conflict_prediction['region']}")
    print(f"Conflict Risk: {conflict_prediction['conflict_risk']:.2%}")
    print(f"Confidence: {conflict_prediction['confidence']:.2f}")
    
    # Demo 4: Disease Outbreak Prediction
    print("\n\n====== DEMO 4: DISEASE OUTBREAK PREDICTION ======")
    # Choose a region for epidemic analysis
    region = "Global"
    epidemic_prediction = await epidemic_weather_engine.predict_epidemic_spread(region)
    
    print(f"Region: {epidemic_prediction['region']}")
    print("Epidemic Predictions:")
    print(f"  - New Cases Trend: {epidemic_prediction['predictions']['new_cases_trend']:.2f}")
    print(f"  - Growth Rate (R): {epidemic_prediction['predictions']['growth_rate']:.2f}")
    print(f"  - Days Until Peak: {epidemic_prediction['predictions']['peak_time_days']}")
    print(f"  - Severity Index: {epidemic_prediction['predictions']['severity_index']:.2f}")
    print(f"  - Containment Effectiveness: {epidemic_prediction['predictions']['containment_effectiveness']:.2f}")
    print(f"Confidence: {epidemic_prediction['confidence']:.2f}")
    
    # Demo 5: Weather Prediction
    print("\n\n====== DEMO 5: EXTREME WEATHER PREDICTION ======")
    # Choose a location for weather prediction
    location = "New York"
    weather_prediction = await epidemic_weather_engine.predict_weather(location, days_ahead=7)
    
    print(f"Location: {weather_prediction['location']}")
    print("Weather Predictions:")
    for day in weather_prediction['predictions']:
        print(f"  {day['date']}: {day['temperature']:.1f}°C, Precipitation: {day['precipitation_prob']:.1%}")
        print(f"  Extreme Event Probability: {day['extreme_event_probability']:.2%}")
    print(f"Confidence: {weather_prediction['confidence']:.2f}")
    
    # Demo 6: Human Decision Prediction
    print("\n\n====== DEMO 6: HUMAN BEHAVIOR PREDICTION ======")
    # Choose a topic for behavior analysis
    topic = "cryptocurrency"
    
    # Fetch social media and news data
    social_data = await behavioral_analyzer.fetch_social_media_data(topic)
    news_data = await behavioral_analyzer.fetch_news_data(topic)
    
    # Predict human behavior
    behavior_prediction = behavioral_analyzer.predict_human_behavior(social_data, news_data, topic)
    
    print(f"Topic: {behavior_prediction['topic']}")
    print(f"Behavior Prediction: {behavior_prediction['prediction']}")
    print(f"Confidence: {behavior_prediction['confidence']:.2f}")
    print("Trend Metrics:")
    print(f"  - Average Sentiment: {behavior_prediction['trend_metrics']['average_sentiment']:.2f}")
    print(f"  - Sentiment Volatility: {behavior_prediction['trend_metrics']['sentiment_volatility']:.2f}")
    
    # Demo 7: Store Prediction on Blockchain
    print("\n\n====== DEMO 7: STORE PREDICTION ON BLOCKCHAIN ======")
    print("This would store predictions on-chain in a real implementation")
    print("Transaction would be signed and sent to the smart contract")
    
    # In a real implementation with proper private key:
    # result = await smart_contract.submit_prediction_to_chain(
    #     "market", 
    #     market_prediction["prediction"], 
    #     market_prediction["confidence"]
    # )
    # print(f"Transaction Hash: {result['transaction_hash']}")
    # print(f"Prediction ID: {result['prediction_id']}")
    
    print("\n\n====== ALL PREDICTIONS COMPLETE ======")
    print("In a full implementation, predictions would be continuously updated")
    print("and validated against real-world outcomes.")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_predictions()) 