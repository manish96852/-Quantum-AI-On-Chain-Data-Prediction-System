import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    print("Trying to import Flask...")
    from flask import Flask
    print("Flask imported successfully!")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return "Hello, World!"
    
    if __name__ == '__main__':
        print("Starting Flask app...")
        app.run(debug=False, host='0.0.0.0', port=5000)
except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure Flask is installed with: pip install flask") 