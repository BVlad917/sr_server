import torch
from flask import Flask, jsonify, request
import numpy as np
import base64
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/super-resolution', methods=['POST'])
def run_image_sr():
    image = request.files['image'].read()
    model_name = request.form['model_name']

    np_arr = np.frombuffer(image, np.uint8)
    img_tensor = torch.from_numpy(np_arr)

    # do stuff with img_tensor
    result_tensor = img_tensor

    np_arr = result_tensor.numpy()
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    _, img_bytes = cv2.imencode('.png', img)
    image_data = base64.b64encode(img_bytes).decode('utf-8')

    response = {'image': image_data}
    return jsonify(response)


@app.route('/ocr', methods=['POST'])
def run_ocr():
    image = request.form['image']
    model_name = request.form['model_name']
    _, image = image.split(',', 1)
    image = base64.b64decode(image)
    np_arr = np.frombuffer(image, np.uint8)
    img_tensor = torch.from_numpy(np_arr)

    # do stuff with img_tensor
    result_text = 'OCR OUTPUT'

    response = {'text': result_text}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
