import tensorflow as tf
import cv2
import numpy as np
from time import perf_counter


def predict_monitor(model, img):
    start = perf_counter()
    img = np.array(img)

    img = img.astype('float32') / 255.0
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x)
    k = np.argmax(preds)
    print(perf_counter() - start)
    if k == 0:
        return "1"
    elif k == 1:
        return "2"
    elif k == 2:
        return "3"
    elif k == 3:
        return "4"

if __name__ == "__main__":
    model = tf.keras.models.load_model('model.h5')

    img = cv2.imread('img.png')

    print(predict_monitor(model, img))

