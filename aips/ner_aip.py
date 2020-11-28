# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   ner_aip.py
 
@Time    :   2020/11/19 8:40 下午
 
@Desc    :   调用ner服务
 
"""

import requests
import json


def get_slot_result(query):
    """
    请求ner服务
    :param query: 用户query
    :return:
    """
    url = 'http://www.huaputu.com:14915/entityparse'
    # url = 'http://192.168.8.183:9015/entityparse'

    # {
    #     "querys": ["奥迪A6多少钱", "我明天下午三点去你家吃饭"],
    #     "fields": [
    #         "general"
    #     ],
    #     "candis": [],
    #     "plugins": ["normlize"],
    #     "mode": "stream"
    # }
    datas = {
        "querys": [query],
        "fields": ["general"],
        "candis": [],
        "plugins": ["normlize"],
        "mode": "stream"
    }

    response = requests.post(
        url,
        data=json.dumps(datas),
        headers={'Content-Type': 'application/json'}
    )
    entities = []
    if response.status_code == 200:
        result = json.loads(response.text.encode('utf8').decode('unicode_escape'))
        entitys = result['result']['entitys']
        if '时间' in entitys[0]:
            for a in entitys[0]['时间']:
                tmp_entity = {'entity': 'date-time',
                              'start': a['locs'][0],
                              'end': a['locs'][1],
                              'value': a['text']}
                entities.append(tmp_entity)
        if '地址名' in entitys[0]:
            for b in entitys[0]['地址名']:
                tmp_entity = {'entity': 'address',
                              'start': b['locs'][0],
                              'end': b['locs'][1],
                              'value': b['text']}
                entities.append(tmp_entity)

        return entities


if __name__ == '__main__':
    query = "明天杭州天气"
    get_slot_result(query)