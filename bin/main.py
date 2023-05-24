from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from predict import load_model, inpainting_image

# List of allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

parameter = load_model();

def allowed_file(filename):
    """
    Check if a file has an allowed file extension
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(config=None):
    app = Flask(__name__)

    # See http://flask.pocoo.org/docs/latest/config/
    app.config.update(dict(DEBUG=True))
    app.config.update(config or {})
    # Configure the static folder to serve processed images
    app.config['STATIC_FOLDER'] = 'processed_images'
    app.static_folder = app.config['STATIC_FOLDER']
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Setup cors headers to allow all domains
    # https://flask-cors.readthedocs.io/en/latest/
    CORS(app)

    # Definition of the routes. Put them into their own file. See also
    # Flask Blueprints: http://flask.pocoo.org/docs/latest/blueprints
    @app.route("/")
    def hello_world():
        return "Hello World"

    @app.route('/process_images', methods=['POST'])
    def process_images():
        try:

            image1 = request.files.get('main')
            image2 = request.files.get('mask')
            images = dict(image=image1, mask=image2);

            # Check if both files are present
            if not image1 or not image2:
                return jsonify({'error': 'Both main and mask files are required.'}), 400

            # Check if the files have allowed extensions
            if not allowed_file(image1.filename) or not allowed_file(image2.filename):
                return jsonify({'error': 'Invalid file format. Allowed formats: JPG, JPEG, PNG, WebP.'}), 400

            # Generate new image using lama inpanting
            filename = inpainting_image(parameter, images);

            # Return the URL of the processed image
            return jsonify({'url': f'http://localhost:8000/static/{filename}.png'})

        except Exception as e:
            # Return a general error message if any error occurs
            print(e)
            return jsonify({'error': 'An error occurred. Please try again later.'}), 500

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app = create_app()
    app.run(host="0.0.0.0", port=port)
