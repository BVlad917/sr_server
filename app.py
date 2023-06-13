from flask import Flask, jsonify, request
import numpy as np
import base64
import cv2
from flask_cors import CORS

from ocr import get_ocr_fn
from sisr import get_sisr_forward_fn

app = Flask(__name__)
CORS(app)


@app.route('/super-resolution', methods=['POST'])
def run_image_sr():
    lr_img_bytes = request.files['image'].read()
    model_name = request.form['model_name']

    lr_img_encoded = np.frombuffer(lr_img_bytes, np.uint8)
    lr_img = cv2.imdecode(lr_img_encoded, cv2.IMREAD_COLOR)

    # process
    sisr_fn = get_sisr_forward_fn(model_name=model_name)
    sr_img = sisr_fn(lr_img)

    _, sr_img_bytes = cv2.imencode('.png', sr_img)
    image_data = base64.b64encode(sr_img_bytes).decode('utf-8')

    response = {'image': image_data}
    return jsonify(response)


@app.route('/ocr', methods=['POST'])
def run_ocr():
    sr_img = request.form['image']
    model_name = request.form['model_name']

    _, image = sr_img.split(',', 1)
    sr_img = base64.b64decode(image)
    sr_img = np.frombuffer(sr_img, np.uint8)
    sr_img = cv2.imdecode(sr_img, cv2.IMREAD_COLOR)

    # process
    ocr_fn = get_ocr_fn(model_name=model_name)
    ocr_output = ocr_fn(sr_img)

    return jsonify(ocr_output)


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
