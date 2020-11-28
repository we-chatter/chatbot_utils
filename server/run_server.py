# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   run_server.py

@Time    :   2020/8/26 2:50 下午

@Desc    :

"""
import datetime
import json
import os
import sys
import logging
from config import CONFIG

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from sanic import Sanic, response
from sanic.response import text, HTTPResponse
from sanic.request import Request
from utils.LogUtils import Logger

from models.intent_core.normal_intent import IntentPredict
from aips.ner_aip import get_slot_result

logger = logging.getLogger(__name__)

app = Sanic(__name__)
app.update_config(CONFIG)

ip = IntentPredict()


@app.route("/")
async def test(request):
    return text('Welcome to the Basic Algrithm platform')


@app.post('/intent')
async def intent_predict(request: Request) -> HTTPResponse:
    """

    :return:
    """
    try:
        querys = request.json["querys"]
        result = json.loads(ip.app(querys))
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res_dic = {
            "result": result,
            "time": localtime
        }
        log_res = json.dumps(res_dic, ensure_ascii=False)
        logger.info(log_res)
        return response.json(res_dic)
    except Exception as e:
        logger.info(e)


@app.post('model/parse')
async def nlu_predict(request: Request) -> HTTPResponse:
    """
    nlu接口， intent + slot拼装返回
    返回的数据结构见resource/nlu输出数据结构.json
    :param request:
    :return:
    """
    try:
        querys = request.json["text"]
        result = json.loads(ip.app([querys]))
        entities = get_slot_result(querys)
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res_dic = {
            "intent": result['intents'][0],
            "intent_ranking": result['intents'],
            "entities": entities
        }
        log_res = json.dumps(res_dic, ensure_ascii=False)
        logger.info(log_res)
        return response.json(res_dic)
    except Exception as e:
        logger.info(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9016, auto_reload=True)
