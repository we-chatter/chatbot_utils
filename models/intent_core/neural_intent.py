# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   neural_intent.py
 
@Time    :   2020/11/10 10:37 上午
 
@Desc    :   初始化并加载onnx意图模型
 
"""

import logging

from models.intent_core.model.config import Config
from models.intent_core.model.intentONNX import IntentModelOnnx

logger = logging.getLogger(__name__)


class NeuralIntentParser(object):
    def __init__(self):
        super(NeuralIntentParser, self).__init__()
        config = Config()
        self.bertClassifier = IntentModelOnnx(config)
        logger.info("==========Intent bert model loaded!==============")

    def predict(self, sentences):
        tagger = self.bertClassifier.predict(sentences)
        return tagger


if __name__=='__main__':
    sentence = ["明天杭州天气"]
    y = NeuralIntentParser().predict(sentence)
    print(y)