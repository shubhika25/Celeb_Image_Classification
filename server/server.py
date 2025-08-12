from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    if 'image_data' not in request.files:
        return jsonify({'error': 'No image file part'}), 400

    image_file = request.files['image_data']
    image_bytes = image_file.read()

    # Call the util method that handles bytes
    result = util.classify_image_bytes(image_bytes)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(host='0.0.0.0', port=5000, debug=True)
