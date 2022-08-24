from abc import ABCMeta, abstractmethod


class BaseAbs(metaclass=ABCMeta):
    def __init__(self, model_path):
        self.model_path = model_path
    
    @abstractmethod
    def _preprocess(self, img):
        pass
    
    @abstractmethod
    def _postprocess(self, output):
        pass
    
    @abstractmethod
    def infer(self, img):
        pass
