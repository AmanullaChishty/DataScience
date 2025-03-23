import joblib
from flask import Flask, request, jsonify
from src.decorators import log_execution

class ModelDeployer:
    """
    Class to simulate model deployment.
    It includes methods to save the trained model and a simple Flask API to serve predictions.
    """
    def __init__(self,model,model_filename="models/energy_consumption_model.joblib"):
        self.model = model
        self.model_filename = model_filename

    @log_execution
    def save_model(self):
        """Save the trained model using joblib."""
        joblib.dump(self.model,self.model_filename)
        return self.model_filename
    
    def load_model(self):
        """Load the trained model from disk."""
        self.model = joblib.load(self.model_filename)
        return self.model

    def run_flask_app(self):
        """Run a simple Flask application to serve predictions."""
        app = Flask(__name__)
        model = self.model

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json(force=True)
            features = data.get('features')
            if features is None:
                return jsonify({"error":"No features provided"}), 400
            prediction = model.predict([features])
            return jsonify({'prediction':prediction.tolist()})

        app.run(debug=True)