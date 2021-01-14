import cv2
import os

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from predict import Predict_gar

# paths to directory
base_dir = ''
caffemodel_path = os.path.join(base_dir, 'models/weights.caffemodel')
prototxt_path = os.path.join(base_dir, 'models/deploy.prototxt')

dir_path = os.path.join(base_dir, 'photo')  # Путь до папки, куда будут заливаться фото
dir_faces = os.path.join(base_dir, 'faces')  # Путь до папки, где будут хранить вырезанные лицы
dir_faces_recognition = os.path.join(base_dir, 'faces_recognition')  # Путь до папки, где будет лежать фоточка с боксами

# load model and initialization predict class
model_multitask = load_model("models/checkpoint_best.h5")
NN_predictor = Predict_gar(model_multitask, base_dir, prototxt_path,
                           caffemodel_path, dir_path, dir_faces, dir_faces_recognition)

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT

    # get, save, read image
    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))
    img = cv2.imread('static/{}.jpg'.format(COUNT))

    # face recognition, save new image with box
    NN_predictor.predict_box_faces(img)
    cv2.imwrite('static/{}_face_rec.jpg'.format(COUNT), img)
    #img.save('static/{}_face_rec.jpg'.format(COUNT))

    # read and cut faces
    img_2 = cv2.imread('static/{}.jpg'.format(COUNT))
    NN_predictor.save_faces(img_2)

    # predict gender, age, race
    text = NN_predictor.return_text_predict()
    text = text.split('\n')[:-1]
    text = ''.join(i + ' ' for i in text)

    COUNT += 1
    return render_template('prediction.html', data=text)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}_face_rec.jpg".format(COUNT - 1))


if __name__ == '__main__':
    app.run(host ='0.0.0.0', port=9091, debug=True)
