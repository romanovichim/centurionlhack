from flask import Flask, request, Response, send_file, jsonify
import numpy as np
import cv2
import base64
import os , io , sys


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




if __name__ == '__main__':
    app.run()
