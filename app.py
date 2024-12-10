from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.relevance import load_model, predict_base_skills
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the model when the application starts
try:
    load_model()  # Ensure this is called to load your trained model
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_skills():
    """Endpoint to predict base skills from input skill variations"""
    try:
        data = request.get_json()
        
        if not data or 'skills' not in data:
            return jsonify({"error": "Invalid request format. Please provide 'skills' array in JSON"}), 400
            
        skills = data['skills']
        
        if not isinstance(skills, list) or not skills:
            return jsonify({"error": "Skills must be provided as a non-empty array"}), 400

        # Make predictions using the imported function
        results = predict_base_skills(skills)

        # Format response
        response = {
            "predictions": [
                {
                    "input_skill": input_skill,
                    "base_skill": base_skill,
                    "confidence": confidence
                }
                for input_skill, base_skill, confidence in results
            ]
        }
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
