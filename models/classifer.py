import time
import numpy as np
import cv2
from abc import abstractmethod
from .base import BaseAbs


class ClassifierAbs(BaseAbs):
    def __init__(self, model_path, input_shape, device):
        super().__init__(model_path)
        self.input_shape = input_shape
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1]).astype(np.float32)
        return img[None]
    
    def _postprocess(self, output):
        cls = np.argmax(output)
        cls_prob = ClassifierAbs.softmax(output)
        return cls, cls_prob
    
    @abstractmethod
    def infer(self, img):
        pass


class ClassiferOnnx(ClassifierAbs):
    def __init__(self, model_path, input_shape, device):
        super().__init__(model_path, input_shape, device)
        import onnxruntime
        print("Start infering using device: %s" % device)
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, img, test_time=1):
        inp = self._preprocess(img)
        speed = list()
        for _ in range(test_time):
            begin = time.time()
            output = self.ort_session.run(None, {self.input_name: inp})[0][0]
            speed.append(time.time()-begin)
        cls, cls_prob = self._postprocess(output)
        return cls, cls_prob, np.mean(speed)

    def infer_batch(self, imgs, test_time=1):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        speed = list()
        for _ in range(test_time):
            begin = time.time()
            np_outputs = self.ort_session.run(None, {self.input_name: inp})[0]
            speed.append(time.time()-begin)
        clss, cls_probs = list(), list()
        for np_output in np_outputs:
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs, np.mean(speed)/len(imgs)
        