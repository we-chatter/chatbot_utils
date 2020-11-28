# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   normal_intent.py
 
@Time    :   2020/11/10 3:47 下午
 
@Desc    :   意图标准化：整合模型和规则提取的结果
 
"""
import json
import time
import logging
import datetime

from models.intent_core.neural_intent import NeuralIntentParser    # 模型

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class IntentPredict(object):

    def __init__(self):
        super(IntentPredict, self).__init__()
        logging.info("===========================意图识别模型开始加载===========================")
        self.predicter = NeuralIntentParser()
        logging.info("===========================意图识别模型加载完成===========================")

    def app(self, querys):
        s = time.time()
        logging.info("======================>查询内容: {}".format(querys))
        res = []
        try:
            results = self.predicter.predict(querys)
            for one_intent, one_prob in results.items():
                tmp_intent = {"name": one_intent, "confidence": one_prob}
                res.append(tmp_intent)
            bodys = {
                "status": "success",
                "code": 200,
                "intents": res[:5],
                "times": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            bodys = {
                "status": "failed",
                "code": 400,
                "entitys": str(e),
                "times": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        e = time.time()
        bodys = json.dumps(bodys, ensure_ascii=False)
        logging.info("======================>查询返回: {}".format(bodys))
        logging.info("======================>查询耗时: {}".format(e - s))
        return bodys