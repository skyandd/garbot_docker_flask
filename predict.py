import os
import cv2
import numpy as np
import shutil

from tensorflow.keras.preprocessing import image as tfimage
from keras_vggface import utils


class Predict_gar:

    def __init__(self, model_multitask, base_dir, prototxt_path, caffemodel_path, dir_path, dir_faces,
                 dir_faces_recognition):

        self.model_multitask = model_multitask
        self.base_dir = base_dir
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.dir_path = dir_path
        self.dir_faces = dir_faces
        self.dir_faces_recognition = dir_faces_recognition
        self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)

    def predict_box_faces(self, image):

        image_dir = os.path.join(self.base_dir, self.dir_faces)
        try:
            os.mkdir(image_dir)
        except:
            pass

        self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.model.setInput(blob)
        detections = self.model.forward()

        # Рисуем боксы
        counter = 0
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]

            if confidence > 0.4:
                counter += 1
                diff = ((endY - startY) - (endX - startX))
                cv2.rectangle(image, (startX - diff, startY - diff // 2), (endX + diff, endY + diff // 2),
                              (255, 255, 255), 2)

                org = (startX - diff, startY - diff // 2 - 10)
                cv2.putText(image, 'person' + str(counter), org,
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        # Записываем картинку в дирректорию
        #cv2.imwrite(os.path.join(image_dir, 'photo.jpg'), image)

    def save_faces(self, image):

        image_dir = os.path.join(self.base_dir, self.dir_faces_recognition)
        try:
            os.mkdir(image_dir)
        except:
            pass

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.model.setInput(blob)
        detections = self.model.forward()

        # Вырезаем лица и сохраняем в дирректорию
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]

            # Поставим порог 0.4, на всякий пожарный)))
            if confidence > 0.5:
                diff = ((endY - startY) - (endX - startX))
                frame = image[startY - diff // 2:endY + diff // 2, startX - diff:endX + diff]
                cv2.imwrite(os.path.join(image_dir, self.dir_faces_recognition + 'person_' + str(i) + 'photo.jpg'),
                            frame)

    def return_text_predict(self):

        gender_mapping = {0: 'Male', 1: 'Female'}
        race_mapping = dict(list(enumerate(('White', 'Black', 'Asian', 'Indian', 'Others'))))
        max_age = 116.0

        counter = 0
        text_output = ''

        for filename in os.listdir(path=os.path.join(self.base_dir, self.dir_faces_recognition)):
            counter += 1
            img = tfimage.load_img(os.path.join(self.base_dir, self.dir_faces_recognition, filename),
                                   target_size=(224, 224))
            x = tfimage.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=2)
            predicted_labels = self.model_multitask.predict(x)
            gender, race, age = int(predicted_labels[0][0] > 0.5), np.argmax(predicted_labels[1][0]), \
                                    predicted_labels[2][0]
            text_output += "Person " + str(
                counter) + f": {gender_mapping[gender]}, {race_mapping[race]}, {int(age[0] * max_age)}" + "\n"

        shutil.rmtree(os.path.join(self.base_dir, self.dir_faces_recognition))
        shutil.rmtree(os.path.join(self.base_dir, self.dir_faces))
        return text_output
