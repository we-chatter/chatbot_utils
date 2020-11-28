# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   intent_classification.py

@Time    :   2020/11/10 3:03 下午

@Desc    :   意图分类模型 transformer + onnx

"""

import os
import sys
import logging
import pandas as pd
import tensorflow as tf
import keras2onnx
from onnxruntime_tools import optimizer
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

sys.path.append('../../../')
from models.intent_core.model.config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

conf = Config()  # 初始化配置文件（超参数配置、数据、预训练资源配置等）


def tf_keras_convert_to_onnx(models, paths, config):
    """
    将keras模型转换为onnx
    :param models:
    :param paths:
    :param config:
    :return:
    """
    onnxNerBert = keras2onnx.convert_keras(
        models,
        models.name,
        target_opset=12
    )
    keras2onnx.save_model(
        onnxNerBert,
        paths
    )
    optimized_model = optimizer.optimize_model(
        paths,
        model_type='bert_keras',
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size
    )
    optimized_model.use_dynamic_axes()
    optimized_model.save_model_to_file(
        paths
    )


def get_labels():
    return sorted(conf.labels, key=conf.labels.index)


def split_dataset(df):
    """
    数据切分
    :param df:
    :return:
    """
    train_set, x = train_test_split(df,
                                    stratify=df['label'],
                                    test_size=0.1,
                                    random_state=42,
                                    shuffle=True)
    val_set, test_set = train_test_split(x,
                                         # stratify=x['label'],  # 确保每一类都被抽样到
                                         test_size=0.5,
                                         random_state=43,
                                         shuffle=True)

    return train_set, val_set, test_set


class OurTokenizer(BertTokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif c.isspace():
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class DataProcess:
    """
    数据预处理
    """


class BertClassifcation(object):
    def __init__(self, config):
        super(BertClassifcation, self).__init__()
        self.pretrain_path = config.pretrain_model_dir
        self.tokenizer = OurTokenizer.from_pretrained(os.path.join(self.pretrain_path, "vocab.txt"),
                                                      do_lower_case=True)
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.learning_rate = config.learn_rate
        self.number_of_epochs = config.epochs
        self.num_classes = config.num_classes  # 类别数
        self.model_path = config.model_path
        self.labels = config.labels

    def convert_example_to_feature(self, review):
        # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
        return self.tokenizer.encode_plus(review,
                                          add_special_tokens=True,  # add [CLS], [SEP]
                                          max_length=self.max_length,  # max length of the text that can go to BERT
                                          pad_to_max_length=True,  # add [PAD] tokens
                                          return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                          truncation=True
                                          )

    def encode_examples(self, ds, limit=-1):
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
            """map to the expected input to TFBertForSequenceClassification, see here"""
            return {
                       "input_ids": input_ids,
                       "token_type_ids": token_type_ids,
                       "attention_mask": attention_masks,
                   }, label

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            ds = ds.take(limit)

        for index, row in ds.iterrows():
            review = row["content"]
            label = row["y"]
            bert_input = self.convert_example_to_feature(review)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

    def model_build(self):
        self.bertConfig = BertConfig.from_pretrained(
            os.path.join(self.pretrain_path, "config.json"),
            num_labels=self.num_classes
        )
        self.model = TFBertForSequenceClassification.from_pretrained(
            os.path.join(self.pretrain_path, "tf_model.h5"),
            config=self.bertConfig
        )
        self.model.summary()

    def compile(self):
        # optimizer Adam recommended
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-08, clipnorm=1)
        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def train(self, dfRaw):
        train_data, val_data, test_data = split_dataset(dfRaw)
        # 数据编码
        # train dataset
        ds_train_encoded = self.encode_examples(train_data).shuffle(10000).batch(
            self.batch_size)
        # val dataset
        ds_val_encoded = self.encode_examples(val_data).batch(self.batch_size)
        # test dataset
        ds_test_encoded = self.encode_examples(test_data).batch(self.batch_size)
        # model initialization
        self.model_build()
        self.compile()
        # fit model
        my_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=3,
                                                 min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max'),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, save_weights_only=True,
                                               monitor='val_accuracy', mode='max', save_best_only=True)
        ]
        bert_history = self.model.fit(ds_train_encoded,
                                      epochs=self.number_of_epochs,
                                      validation_data=ds_val_encoded,
                                      verbose=2,
                                      callbacks=my_callbacks
                                      )
        # res = self.model.predict(["我爱你"])
        # evaluate test_set
        logging.info("# evaluate test_set:", self.model.evaluate(ds_test_encoded))
        tf_keras_convert_to_onnx(
            self.model,
            self.model_path,
            self.bertConfig
        )

    def predict(self, sentences):
        features = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="tf",
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )

def app():
    customerData = pd.read_csv(conf.data_path, header=0, sep='\t').dropna(axis=0, inplace=False)
    logging.info(customerData.head())
    dfLables = pd.DataFrame({"label": conf.labels, "y": list(range(conf.num_classes))})
    dfRaw = pd.merge(customerData, dfLables, on="label", how="left")
    logging.info(dfRaw.head())
    bertModel = BertClassifcation(conf)
    bertModel.train(dfRaw)


if __name__ == '__main__':
    app()
