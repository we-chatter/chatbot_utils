# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   mysql_utils.py
 
@Time    :   2020/8/11 6:57 下午
 
@Desc    :   连接mysql数据库工具
 
"""

import pymysql
import logging

logger = logging.getLogger(__name__)


class IntentDetail:
    def __init__(self):
        # 本地
        self.database = 'renren_security'
        self.host = 'rm-uf6rv6pnfqsf649w92o.mysql.rds.aliyuncs.com'
        self.username = 'gpu'
        self.password = 'kvqpZxG6'
        self.db = pymysql.connect(host=self.host, port=3306, user=self.username, passwd=self.password, db=self.database)

    def get_data(self):
        """
        根据技能Id获取其下面的所有意图详情数据
        :param skillId:
        :return:
        """
        cursor = self.db.cursor()
        # cursor.execute("select robot_intention_purpose_id, intent_name from data_center_dialogue_log where skill_id = %s", skillId)
        cursor.execute(
            "select b.id,b.purpose from robot_intention_purpose_desc a left join robot_intention_purpose b on  a.robot_intention_purpose_id=b.id ")
        res = cursor.fetchall()
        logger.info('All data length is {}'.format(len(res)))
        cursor.close()
        self.db.close()

        return res


if __name__ == '__main__':
    intent = IntentDetail().get_data()
