# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   config.py

@Time    :   2020/11/10 3:03 下午

@Desc    :   意图识别参数配置文件

"""

import os
import codecs
import logging
import pathlib

basedir=str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class Config(object):
    def __init__(self):
        self.max_length = 64
        self.learn_rate = 2e-5
        self.batch_size = 16
        self.epochs = 5
        self.model_path = basedir + '/models/intent_core/saved_models/intent_core.onnx'
        self.data_path = basedir + '/data/intention/train.txt'
        self.labels = [c.strip().split('\t')[0] for c in codecs.open(self.data_path, 'r', encoding='utf-8').readlines()]
        self.labels = sorted(list(set(self.labels)), key=self.labels.index)  # 要求label的顺序保持一致
        self.labels.remove('label')
        logging.info("label is {}".format(self.labels))
        self.num_classes = len(self.labels)
        self.labels2id = {label: index for index, label in enumerate(self.labels)}
        self.trained = True
        # self.pretrain_model_dir = '/data/pretrain/tensorflow2.x/chinese-roberta-wwm-ext-large'
        self.pretrain_model_dir = '/Data/public/pretrained_models/tensorflow2.x/chinese-roberta-wwm-ext'