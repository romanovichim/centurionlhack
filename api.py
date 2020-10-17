from flask import Flask, request, Response, send_file, jsonify
import numpy as np
import cv2
import base64
import os , io , sys
from facepixel import pixelfaces
from parkingdetection import apigenerateparkboxes,apidetectparking


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Initialize the Flask application
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# route http posts to this method
#For test take image returns image
@app.route('/api/test', methods=['POST'])
def test():
    # print(request.files , file=sys.stderr)
    file = request.files['image'] ## byte file
    if file and allowed_file(file.filename):
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #image transformation here
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        return jsonify({'status': to_send})

    else:
        return jsonify({'error': 'not allowed extension'}), 400

#Blur face for legal
@app.route('/api/blurface', methods=['POST'])
def blurface():
    # print(request.files , file=sys.stderr)
    file = request.files['image'] ## byte file
    if file and allowed_file(file.filename):
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #image transformation here
        image = pixelfaces(image)
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        return jsonify({'status': to_send})

    else:
        return jsonify({'error': 'not allowed extension'}), 400

#Detect parking 
@app.route('/api/apidetectparking', methods=['POST'])
def apidetectparking():
    # print(request.files , file=sys.stderr)
    file = request.files['image'] ## byte file
    if file and allowed_file(file.filename):
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #permanenttoken hardcode - later done
        token = list123
        apigenerateparkboxes(image,token)
        #image transformation here
        image = pixelfaces(image)
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        return jsonify({'status': to_send})

    else:
        return jsonify({'error': 'smth goes wrong'}), 400

#Detect space
@app.route('/api/apidetectspace', methods=['POST'])
def apidetectspace():
    # print(request.files , file=sys.stderr)
    file = request.files['image'] ## byte file
    if file and allowed_file(file.filename):
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #image transformation here
        #token next time
        apidetectparking(imagename)
        image = pixelfaces(image)
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        return jsonify({'status': to_send})

    else:
        return jsonify({'error': 'not allowed extension'}), 400





if __name__ == '__main__':
    app.run()
