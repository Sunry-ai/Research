import cv2
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import numpy as np
import base64
import sklearn
"""
確認用のプログラムです。本番のpyにアサインする際に要変更です。
"""


class svm_model(BaseEstimator,ClassifierMixin):
    def __init__(self):
        """
        モデルのpathは　MNIST_SVMですが、本番のpyにアサインする際に、pathは要変更です。
        """
        # Load Model
        self.model=joblib.load("MNIST_SVM")

    def predict(self,x):
        """
        xはinput画像です。ただし、base64でエンコードされた画像です。
        inferする際に、ここのxがリストになるので、x[0]がbase64でエンコードされた画像になることを気をつけてください。
        """
        str_decode = base64.b64decode(x)
        nparr=np.frombuffer(str_decode, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        result = []
        img = np.reshape(img, (1, 64))[0]
        img = img.reshape(1, -1)
        number = str(self.model.predict(img)[0])
        result.append({"number": number})

        return result

model_a=svm_model()
print(model_a.predict(b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAIAAgBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APwRDkaUqusfkeQcEgZMpb884/Sv/9k='))