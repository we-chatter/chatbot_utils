# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   intentONNX.py

@Time    :   2020/11/10 3:03 下午

@Desc    :   加载onnx模型

"""

import os
import onnxruntime
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from models.intent_core.model.config import Config

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def batch_generater(features, batch_size=128):
    input_ids = features["input_ids"].numpy()
    token_type_ids = features["token_type_ids"].numpy()
    attention_mask = features["attention_mask"].numpy()

    if len(input_ids) % batch_size == 0:
        n = len(input_ids) // batch_size
    else:
        n = len(input_ids) // batch_size + 1

    for j in range(n):
        yield {
            "input_ids": input_ids[j * batch_size:(j + 1) * batch_size],
            "token_type_ids": token_type_ids[j * batch_size:(j + 1) * batch_size],
            "attention_mask": attention_mask[j * batch_size:(j + 1) * batch_size]
        }


class ourTokenizer(BertTokenizer):
    def _tokenize(self, text):
        tokens = []
        for curPiece in text:
            if curPiece in self.vocab:
                tokens.append(curPiece)
            elif curPiece.isspace():
                tokens.append("[unused1]")
            else:
                tokens.append("[UNK]")
        return tokens


class IntentModelOnnx(object):
    def __init__(self, config):
        super(IntentModelOnnx, self).__init__()
        self.model_path = config.model_path
        self.max_seq_len = config.max_length
        self.pretrain_model_path = config.pretrain_model_dir
        self.labels = config.labels
        self.tokenizer = ourTokenizer.from_pretrained(
            os.path.join(self.pretrain_model_path, "vocab.txt"),
            do_lower_case=True
        )
        self.sess_options = onnxruntime.SessionOptions()
        self.nerBertONNX = onnxruntime.InferenceSession(
            self.model_path,
            self.sess_options,
            providers=["CUDAExecutionProvider"]
        )

    def predict(self, sentences):
        features = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="tf",
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )
        batchs = batch_generater(features)
        logits = np.vstack([self.nerBertONNX.run(None, b)[0] for b in batchs])
        probs = tf.nn.softmax(logits, axis=-1).numpy().tolist()[0]
        # y = np.argmax(probs, axis=1)  # [np.argmax(pred) for pred in y_pred]
        # y_labels = [self.labels[Y] for Y in y.tolist()]
        intent = dict(zip(self.labels, probs))
        print(intent)
        intent = dict(sorted(intent.items(), key=lambda e: e[1], reverse=True))
        return intent


def app():
    import time
    salesmanS = ['我的银行卡丢了']
    salesmanConfig = Config()
    modelONNX = IntentModelOnnx(salesmanConfig)
    s_t = time.time()
    res = modelONNX.predict(salesmanS)
    print("耗时: {:.3f}".format(time.time() - s_t))
    print(res)


if __name__ == '__main__':
    app()
