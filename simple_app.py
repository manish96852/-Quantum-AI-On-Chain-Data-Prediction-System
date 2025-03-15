from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Quantum AI + On-Chain Data Prediction System is online!"

@app.route('/api/test')
def test():
    return jsonify({"status": "success", "message": "API is working!"})

if __name__ == '__main__':
    print("Starting simple test app on port 5000...")
    app.run(debug=False, host='0.0.0.0', port=5000) 