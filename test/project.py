#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from tensorflow.keras.layers import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras import optimizers
from keras.models import load_model
import pymysql, urllib.request, csv, re, datetime
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dateutil.relativedelta import relativedelta

#基隆市KEL Keelung
KEL_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%9F%BA%E9%9A%86%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#新北市NTPC New_Taipei
NTPC_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E6%96%B0%E5%8C%97%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#台北市TPE Taipei
TPE_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%8F%B0%E5%8C%97%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#桃園市TYN Taoyuan
TYN_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E6%A1%83%E5%9C%92%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#新竹縣HSZ0 Hsinchu_County
HSZ0_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E6%96%B0%E7%AB%B9%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#新竹市HSZ1 Hsinchu_City
HSZ1_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E6%96%B0%E7%AB%B9%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#苗栗縣ZMI Miaoli
ZMI_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E8%8B%97%E6%A0%97%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#台中市TXG Taichung
TXG_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%8F%B0%E4%B8%AD%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#彰化縣CHW Changhua
CHW_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%BD%B0%E5%8C%96%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#南投縣NTC Nantou
NTC_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%8D%97%E6%8A%95%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#雲林縣YUN Yunlin
YUN_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E9%9B%B2%E6%9E%97%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#嘉義縣CYI0 Chiayi_County
CYI0_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%98%89%E7%BE%A9%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#嘉義市CYI1 Chiayi_City
CYI1_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%98%89%E7%BE%A9%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#台南市TNN Tainan
TNN_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%8F%B0%E5%8D%97%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#高雄市KHH Kaohsiung
KHH_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E9%AB%98%E9%9B%84%E5%B8%82%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#屏東縣PIF Pingtung
PIF_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%B1%8F%E6%9D%B1%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#宜蘭縣ILA Yilan
ILA_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%AE%9C%E8%98%AD%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#花蓮縣HUN Hualien
HUN_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E8%8A%B1%E8%93%AE%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#台東縣TTT Taitung
TTT_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E5%8F%B0%E6%9D%B1%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#澎湖縣PEH Penghu
PEH_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E6%BE%8E%E6%B9%96%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#金門縣KNH Kinmen
KNH_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E9%87%91%E9%96%80%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"
#連江縣LNN Lienchiang
LNN_url = "https://raw.githubusercontent.com/lintao89/project_data/main/%E9%80%A3%E6%B1%9F%E7%B8%A3%E6%AF%8F%E6%97%A5%E7%A2%BA%E8%A8%BA%E6%95%B8.csv"

plt.switch_backend('agg')

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create():
    conn = pymysql.connect(
        host = 'localhost',
        user = 'root',
        password = '0000',
        database = '專題'
    )
    cursor = conn.cursor()
    
    sql_create = '''
    CREATE TABLE IF NOT EXISTS Taiwan (
    個案研判日 Date NULL, 
    縣市 VARCHAR(255) NULL, 
    確定病例數 VARCHAR(255) NULL);'''
    cursor.execute(sql_create)
    cursor.execute('truncate Taiwan;') #清空資料表
    
    #台中市TXG Taichung
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(TXG_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()

    #基隆市KEL Keelung
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(KEL_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #新北市NTPC New_Taipei
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(NTPC_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #台北市TPE Taipei
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(TPE_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #桃園市TYN Taoyuan
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(TYN_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #新竹縣HSZ0 Hsinchu_County
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(HSZ0_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #新竹市HSZ1 Hsinchu_City
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(HSZ1_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #苗栗縣ZMI Miaoli
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(ZMI_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #彰化縣CHW Changhua
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(CHW_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #南投縣NTC Nantou
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(NTC_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #雲林縣YUN Yunlin
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(YUN_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #嘉義縣CYI0 Chiayi_County
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(CYI0_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #嘉義市CYI1 Chiayi_City
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(CYI1_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #台南市TNN Tainan
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(TNN_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #高雄市KHH Kaohsiung
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(KHH_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #屏東縣PIF Pingtung
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(PIF_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #宜蘭縣ILA Yilan
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(ILA_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #花蓮縣HUN Hualien
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(HUN_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #台東縣TTT Taitung
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(TTT_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #澎湖縣PEH Penghu
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(PEH_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #金門縣KNH Kinmen
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(KNH_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #連江縣LNN Lienchiang
    sql_insert = '''insert into Taiwan value(%s, %s, %s);'''
    with urllib.request.urlopen(LNN_url) as f:
        f.readline().decode('UTF-8')
        while (True):
            res = f.readline().strip().decode(encoding='UTF-8').split(',')
            # 到沒資料離開迴圈
            if res != ['']:
                cursor.execute(sql_insert,[res[0], res[1], res[2]])
                #print(res)
            else:
                break
        conn.commit()
    
    #cursor.close()
    #conn.close()

create()


# In[ ]:


# 初始化sqlalchemy
db = SQLAlchemy()
app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:0000@localhost:3306/專題"
# [DB_TYPE]+[DB_CONNECTOR]://[USERNAME]:[PASSWORD]@[HOST]:[PORT]/[DB_NAME]

CORS(app)
db.init_app(app)
#db = SQLAlchemy(app)

first_url = "https://raw.githubusercontent.com/zu-z/project/main/first_web.html"
#"https://htmlpreview.github.io/?https://github.com/zu-z/project/blob/main/first_web.html"
second_url = "https://raw.githubusercontent.com/zu-z/project/main/second_web.html"
#"https://htmlpreview.github.io/?https://github.com/zu-z/project/blob/main/second_web.html"
third_url = "https://raw.githubusercontent.com/zu-z/project/main/third_web.html"
#"https://htmlpreview.github.io/?https://github.com/zu-z/project/blob/main/third_web.html"

@app.route("/")
def first_web():
    #with urllib.request.urlopen(first_url) as response:
        #first = response.read()
    #return first
    return render_template('first_web.html')

@app.route("/second_web", methods=["POST","GET"])
def second_web():
    print(request.method)
    if request.method == "POST":
        user_input = request.values['user_input']
        user_date = request.values['user_date']
    else:
        user_input = request.args.get('user_input')
        user_date = request.args.get('user_date')
    
    date = datetime.datetime.strptime(user_date, "%Y-%m-%d")
    
    if(user_input == '一天'):
        user_date_ = user_date
        date_all = []
        date_all.append(user_date_)
        
        #基隆市KEL Keelung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "基隆市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KEL = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KEL) == 0):
            data_all_KEL.append(('0',))
        
        #新北市NTPC New_Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新北市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTPC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTPC) == 0):
            data_all_NTPC.append(('0',))
        
        #台北市TPE Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台北市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TPE = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TPE) == 0):
            data_all_TPE.append(('0',))
        
        #桃園市TYN Taoyuan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "桃園市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TYN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TYN) == 0):
            data_all_TYN.append(('0',))
        
        #新竹縣HSZ0 Hsinchu_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ0) == 0):
            data_all_HSZ0.append(('0',))
        
        #新竹市HSZ1 Hsinchu_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ1) == 0):
            data_all_HSZ1.append(('0',))
        
        #苗栗縣ZMI Miaoli
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "苗栗縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ZMI = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ZMI) == 0):
            data_all_ZMI.append(('0',))
        
        #台中市TXG Taichung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台中市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TXG = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TXG) == 0):
            data_all_TXG.append(('0',))
        
        #彰化縣CHW Changhua
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "彰化縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CHW = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CHW) == 0):
            data_all_CHW.append(('0',))
        
        #南投縣NTC Nantou
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "南投縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTC) == 0):
            data_all_NTC.append(('0',))
        
        #雲林縣YUN Yunlin
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "雲林縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_YUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_YUN) == 0):
            data_all_YUN.append(('0',))
        
        #嘉義縣CYI0 Chiayi_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI0) == 0):
            data_all_CYI0.append(('0',))
        
        #嘉義市CYI1 Chiayi_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI1) == 0):
            data_all_CYI1.append(('0',))
        
        #台南市TNN Tainan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台南市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TNN) == 0):
            data_all_TNN.append(('0',))
        
        #高雄市KHH Kaohsiung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "高雄市" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KHH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KHH) == 0):
            data_all_KHH.append(('0',))
        
        #屏東縣PIF Pingtung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "屏東縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PIF = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PIF) == 0):
            data_all_PIF.append(('0',))
        
        #宜蘭縣ILA Yilan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "宜蘭縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ILA = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ILA) == 0):
            data_all_ILA.append(('0',))
        
        #花蓮縣HUN Hualien
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "花蓮縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HUN) == 0):
            data_all_HUN.append(('0',))
        
        #台東縣TTT Taitung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台東縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TTT = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TTT) == 0):
            data_all_TTT.append(('0',))
        
        #澎湖縣PEH Penghu
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "澎湖縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PEH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PEH) == 0):
            data_all_PEH.append(('0',))
        
        #金門縣KNH Kinmen
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "金門縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KNH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KNH) == 0):
            data_all_KNH.append(('0',))
        
        #連江縣LNN Lienchiang
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "連江縣" AND 個案研判日 = "%s" ;""" %(user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_LNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_LNN) == 0):
            data_all_LNN.append(('0',))

    elif(user_input == '一周'):
        user_date_ = (date + datetime.timedelta(days=6)).strftime("%Y-%m-%d")
        date_all = []
        for i in range(7):
            date_all += ((date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")).split()
        print(date_all)
        print(type(date_all))
        patch_zero = (('0',),('0',),('0',),('0',),('0',),('0',),('0',))
        
        #基隆市KEL Keelung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "基隆市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KEL = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KEL) == 0):
            data_all_KEL.extend(patch_zero)
        
        #新北市NTPC New_Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新北市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTPC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTPC) == 0):
            data_all_NTPC.extend(patch_zero)
        
        #台北市TPE Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台北市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TPE = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TPE) == 0):
            data_all_TPE.extend(patch_zero)
        
        #桃園市TYN Taoyuan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "桃園市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TYN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TYN) == 0):
            data_all_TYN.extend(patch_zero)
        
        #新竹縣HSZ0 Hsinchu_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ0) == 0):
            data_all_HSZ0.extend(patch_zero)
        
        #新竹市HSZ1 Hsinchu_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ1) == 0):
            data_all_HSZ1.extend(patch_zero)
        
        #苗栗縣ZMI Miaoli
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "苗栗縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ZMI = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ZMI) == 0):
            data_all_ZMI.extend(patch_zero)
        
        #台中市TXG Taichung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台中市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TXG = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TXG) == 0):
            data_all_TXG.extend(patch_zero)
        
        #彰化縣CHW Changhua
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "彰化縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CHW = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CHW) == 0):
            data_all_CHW.extend(patch_zero)
        
        #南投縣NTC Nantou
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "南投縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTC) == 0):
            data_all_NTC.extend(patch_zero)
        
        #雲林縣YUN Yunlin
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "雲林縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_YUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_YUN) == 0):
            data_all_YUN.extend(patch_zero)
        
        #嘉義縣CYI0 Chiayi_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI0) == 0):
            data_all_CYI0.extend(patch_zero)
        
        #嘉義市CYI1 Chiayi_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI1) == 0):
            data_all_CYI1.extend(patch_zero)
        
        #台南市TNN Tainan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台南市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TNN) == 0):
            data_all_TNN.extend(patch_zero)
        
        #高雄市KHH Kaohsiung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "高雄市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KHH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KHH) == 0):
            data_all_KHH.extend(patch_zero)
        
        #屏東縣PIF Pingtung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "屏東縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PIF = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PIF) == 0):
            data_all_PIF.extend(patch_zero)
        
        #宜蘭縣ILA Yilan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "宜蘭縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ILA = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ILA) == 0):
            data_all_ILA.extend(patch_zero)
        
        #花蓮縣HUN Hualien
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "花蓮縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HUN) == 0):
            data_all_HUN.extend(patch_zero)
        
        #台東縣TTT Taitung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台東縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TTT = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TTT) == 0):
            data_all_TTT.extend(patch_zero)
        
        #澎湖縣PEH Penghu
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "澎湖縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PEH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PEH) == 0):
            data_all_PEH.extend(patch_zero)
        
        #金門縣KNH Kinmen
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "金門縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KNH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KNH) == 0):
            data_all_KNH.extend(patch_zero)
        
        #連江縣LNN Lienchiang
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "連江縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_LNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_LNN) == 0):
            data_all_LNN.extend(patch_zero)
            
    elif(user_input == '一月'):
        user_date_ = (date + relativedelta(months = +1)).strftime("%Y-%m-%d")
        d_start = datetime.datetime.strptime(user_date, "%Y-%m-%d")
        d_end = datetime.datetime.strptime(user_date_, "%Y-%m-%d")
        d_delta = d_end - d_start
        date_all = []
        patch_zero = []
        for i in range(d_delta.days+1):
            date_all += ((date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")).split()
            patch_zero.append(('0',))
        print(date_all)
        print(type(date_all))
        
        #基隆市KEL Keelung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "基隆市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KEL = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KEL) == 0):
            data_all_KEL.extend(patch_zero)
        
        #新北市NTPC New_Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新北市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTPC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTPC) == 0):
            data_all_NTPC.extend(patch_zero)
        
        #台北市TPE Taipei
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台北市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TPE = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TPE) == 0):
            data_all_TPE.extend(patch_zero)
        
        #桃園市TYN Taoyuan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "桃園市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TYN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TYN) == 0):
            data_all_TYN.extend(patch_zero)
        
        #新竹縣HSZ0 Hsinchu_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ0) == 0):
            data_all_HSZ0.extend(patch_zero)
        
        #新竹市HSZ1 Hsinchu_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "新竹市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HSZ1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HSZ1) == 0):
            data_all_HSZ1.extend(patch_zero)
        
        #苗栗縣ZMI Miaoli
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "苗栗縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ZMI = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ZMI) == 0):
            data_all_ZMI.extend(patch_zero)
        
        #台中市TXG Taichung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台中市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TXG = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TXG) == 0):
            data_all_TXG.extend(patch_zero)
        
        #彰化縣CHW Changhua
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "彰化縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CHW = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CHW) == 0):
            data_all_CHW.extend(patch_zero)
        
        #南投縣NTC Nantou
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "南投縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_NTC = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_NTC) == 0):
            data_all_NTC.extend(patch_zero)
        
        #雲林縣YUN Yunlin
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "雲林縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_YUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_YUN) == 0):
            data_all_YUN.extend(patch_zero)
        
        #嘉義縣CYI0 Chiayi_County
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI0 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI0) == 0):
            data_all_CYI0.extend(patch_zero)
        
        #嘉義市CYI1 Chiayi_City
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "嘉義市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_CYI1 = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_CYI1) == 0):
            data_all_CYI1.extend(patch_zero)
        
        #台南市TNN Tainan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台南市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TNN) == 0):
            data_all_TNN.extend(patch_zero)
        
        #高雄市KHH Kaohsiung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "高雄市" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KHH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KHH) == 0):
            data_all_KHH.extend(patch_zero)
        
        #屏東縣PIF Pingtung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "屏東縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PIF = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PIF) == 0):
            data_all_PIF.extend(patch_zero)
        
        #宜蘭縣ILA Yilan
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "宜蘭縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_ILA = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_ILA) == 0):
            data_all_ILA.extend(patch_zero)
        
        #花蓮縣HUN Hualien
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "花蓮縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_HUN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_HUN) == 0):
            data_all_HUN.extend(patch_zero)
        
        #台東縣TTT Taitung
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "台東縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_TTT = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_TTT) == 0):
            data_all_TTT.extend(patch_zero)
        
        #澎湖縣PEH Penghu
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "澎湖縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_PEH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_PEH) == 0):
            data_all_PEH.extend(patch_zero)
        
        #金門縣KNH Kinmen
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "金門縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_KNH = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_KNH) == 0):
            data_all_KNH.extend(patch_zero)
        
        #連江縣LNN Lienchiang
        sql = """ SELECT 確定病例數 from Taiwan where 縣市 = "連江縣" 
        AND 個案研判日 >= "%s" AND 個案研判日 <= "%s" ;""" %(user_date, user_date_)
        db.engine.execute(sql) #執行 SQL 指令
        data_all_LNN = db.engine.execute(sql).fetchall() #取出全部資料
        if (len(data_all_LNN) == 0):
            data_all_LNN.extend(patch_zero)
        
    
    # list轉string後，取re數字(list)
    data_re01 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_KEL))
    data_re02 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_NTPC))
    data_re03 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_TPE))
    data_re04 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_TYN))
    data_re05 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_HSZ0))
    data_re06 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_HSZ1))
    data_re07 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_ZMI))
    data_re08 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_TXG))
    data_re09 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_CHW))
    data_re10 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_NTC))
    data_re11 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_YUN))
    data_re12 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_CYI0))
    data_re13 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_CYI1))
    data_re14 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_TNN))
    data_re15 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_KHH))
    data_re16 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_PIF))
    data_re17 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_ILA))
    data_re18 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_HUN))
    data_re19 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_TTT))
    data_re20 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_PEH))
    data_re21 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_KNH))
    data_re22 = re.findall(r"\d+", ",".join("%s" %id for id in data_all_LNN))
    
    #用彰化data_re09來判斷list長度後，前面項目補0
    for i in range(len(data_re09)-len(data_re01)):
        data_re01.insert(0, '0')
    for i in range(len(data_re09)-len(data_re02)):
        data_re02.insert(0, '0')
    for i in range(len(data_re09)-len(data_re03)):
        data_re03.insert(0, '0')
    for i in range(len(data_re09)-len(data_re04)):
        data_re04.insert(0, '0')
    for i in range(len(data_re09)-len(data_re05)):
        data_re05.insert(0, '0')
    for i in range(len(data_re09)-len(data_re06)):
        data_re06.insert(0, '0')
    for i in range(len(data_re09)-len(data_re07)):
        data_re07.insert(0, '0')
    for i in range(len(data_re09)-len(data_re08)):
        data_re08.insert(0, '0')
    for i in range(len(data_re09)-len(data_re10)):
        data_re10.insert(0, '0')
    for i in range(len(data_re09)-len(data_re11)):
        data_re11.insert(0, '0')
    for i in range(len(data_re09)-len(data_re12)):
        data_re12.insert(0, '0')
    for i in range(len(data_re09)-len(data_re13)):
        data_re13.insert(0, '0')
    for i in range(len(data_re09)-len(data_re14)):
        data_re14.insert(0, '0')
    for i in range(len(data_re09)-len(data_re15)):
        data_re15.insert(0, '0')
    for i in range(len(data_re09)-len(data_re16)):
        data_re16.insert(0, '0')
    for i in range(len(data_re09)-len(data_re17)):
        data_re17.insert(0, '0')
    for i in range(len(data_re09)-len(data_re18)):
        data_re18.insert(0, '0')
    for i in range(len(data_re09)-len(data_re19)):
        data_re19.insert(0, '0')
    for i in range(len(data_re09)-len(data_re20)):
        data_re20.insert(0, '0')
    for i in range(len(data_re09)-len(data_re21)):
        data_re21.insert(0, '0')
    for i in range(len(data_re09)-len(data_re22)):
        data_re22.insert(0, '0')
    
    # 全國人數加總list
    data_all = []
    for i in range(len(data_re09)): 
        total = (int(data_re01[i]) + int(data_re02[i]) + int(data_re03[i]) + int(data_re04[i]) + 
                 int(data_re05[i]) + int(data_re06[i]) + int(data_re07[i]) + int(data_re08[i]) + 
                 int(data_re09[i]) + int(data_re10[i]) + int(data_re11[i]) + int(data_re12[i]) + 
                 int(data_re13[i]) + int(data_re14[i]) + int(data_re15[i]) + int(data_re16[i]) + 
                 int(data_re17[i]) + int(data_re18[i]) + int(data_re19[i]) + int(data_re20[i]) + 
                 int(data_re21[i]) + int(data_re22[i]))
        print(total)
        data_all.append(total)
    print(data_all)
    
    # 日期最後一天的確診人數
    today_data = data_all[-1]
    data_KEL = data_re01[-1]
    data_NTPC = data_re02[-1]
    data_TPE = data_re03[-1]
    data_TYN = data_re04[-1]
    data_HSZ0 = data_re05[-1]
    data_HSZ1 = data_re06[-1]
    data_ZMI = data_re07[-1]
    data_TXG = data_re08[-1]
    data_CHW = data_re09[-1]
    data_NTC = data_re10[-1]
    data_YUN = data_re11[-1]
    data_CYI0 = data_re12[-1]
    data_CYI1 = data_re13[-1]
    data_TNN = data_re14[-1]
    data_KHH = data_re15[-1]
    data_PIF = data_re16[-1]
    data_ILA = data_re17[-1]
    data_HUN = data_re18[-1]
    data_TTT = data_re19[-1]
    data_PEH = data_re20[-1]
    data_KNH = data_re21[-1]
    data_LNN = data_re22[-1]
    
    # 合併list，全部日期 + 全國確診人數
    list_all = dict(zip(date_all, data_all))
    
    print("user_input：", user_input, type(user_input)) #選擇幾天
    print("user_date：", user_date, type(user_date)) #選擇的日期
    print("user_date_：", user_date_, type(user_date_)) #相加後的日期
    print("data_all_TPE：", data_all_TPE, type(data_all_TPE)) #執行sql指令後的data
    print("data_re03：", data_re03, type(data_re03)) #list轉string後，取re數字(list)
    print("data_all：", data_all, type(data_all)) #全國人數加總list
    print("today_data：", today_data, type(today_data)) #日期最後一天的全國確診人數
    print("data_TPE：", data_TPE, type(data_TPE)) #日期最後一天的TPE確診人數
    print("list_all：", list_all, type(list_all)) #合併list
    
    #with urllib.request.urlopen(second_url) as response:
        #second = response.read()
    #return second
    return render_template("second_web.html", **locals())

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = request.values['user_input']
    '''if(user_input == '一天'):
        #print(date_all)
        #print(type(date_all))
        #Load the trained model. (Pickle file)
        model = load_model('models/taichung_rnn+lstm_1000_1.h5')
        ######################################台北市######################################
	    # 載入訓練資料
        model_TPE = load_model('models/taipei_rnn+lstm_1000_1.h5')
        dataframe_TPE = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TPE=dataframe_TPE.drop(dataframe_TPE[dataframe_TPE['確定病例數']==0].index,axis=0)
        dataset = dataframe_TPE.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TPE_Predict = model_TPE.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TPE_Predict = scaler.inverse_transform(TPE_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TPE_PredictPlot = numpy.empty_like(dataset)
        TPE_PredictPlot[:, :] = numpy.nan
        TPE_PredictPlot[look_back:len(TPE_Predict)+look_back, :] = TPE_Predict
        plt.plot(TPE_PredictPlot)
        plt.savefig('./static/TPE_plot.png') 
        plt.clf()
        #################################################################################
        ######################################基隆市######################################
        dataframe_KEL = pandas.read_csv(r'D:\專題\各縣市每日確診資料\基隆市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KEL=dataframe_KEL.drop(dataframe_KEL[dataframe_KEL['確定病例數']==0].index,axis=0)
        dataset = dataframe_KEL.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KEL_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KEL_Predict = scaler.inverse_transform(KEL_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    #shift train predictions for plotting
        KEL_PredictPlot = numpy.empty_like(dataset)
        KEL_PredictPlot[:, :] = numpy.nan
        KEL_PredictPlot[look_back:len(KEL_Predict)+look_back, :] = KEL_Predict
        plt.plot(KEL_PredictPlot)
        plt.savefig('KEL_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新北市######################################
        dataframe_NTPC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTPC=dataframe_NTPC.drop(dataframe_NTPC[dataframe_NTPC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTPC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTPC_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTPC_Predict = scaler.inverse_transform(NTPC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTPC_PredictPlot = numpy.empty_like(dataset)
        NTPC_PredictPlot[:, :] = numpy.nan
        NTPC_PredictPlot[look_back:len(NTPC_Predict)+look_back, :] = NTPC_Predict
        plt.plot(NTPC_PredictPlot)
        plt.savefig('NTPC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新竹市######################################
        dataframe_HSZ1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ1=dataframe_HSZ1.drop(dataframe_HSZ1[dataframe_HSZ1['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ1_Predict = scaler.inverse_transform(HSZ1_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ1_PredictPlot = numpy.empty_like(dataset)
        HSZ1_PredictPlot[:, :] = numpy.nan
        HSZ1_PredictPlot[look_back:len(HSZ1_Predict)+look_back, :] = HSZ1_Predict
        plt.plot(HSZ1_PredictPlot)
        plt.savefig('HSZ1_plot.png')
        plt.clf()
	    #################################################################################
        ######################################新竹縣######################################
        dataframe_HSZ0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ0=dataframe_HSZ0.drop(dataframe_HSZ0[dataframe_HSZ0['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ0_Predict = scaler.inverse_transform(HSZ0_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ0_PredictPlot = numpy.empty_like(dataset)
        HSZ0_PredictPlot[:, :] = numpy.nan
        HSZ0_PredictPlot[look_back:len(HSZ0_Predict)+look_back, :] = HSZ0_Predict
        plt.plot(HSZ0_PredictPlot)
        plt.savefig('HSZ0_plot.png')
        plt.clf()
	    #################################################################################
        ######################################苗栗縣######################################
        dataframe_ZMI = pandas.read_csv(r'D:\專題\各縣市每日確診資料\苗栗縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ZMI=dataframe_ZMI.drop(dataframe_ZMI[dataframe_ZMI['確定病例數']==0].index,axis=0)
        dataset = dataframe_ZMI.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ZMI_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ZMI_Predict = scaler.inverse_transform(ZMI_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ZMI_PredictPlot = numpy.empty_like(dataset)
        ZMI_PredictPlot[:, :] = numpy.nan
        ZMI_PredictPlot[look_back:len(ZMI_Predict)+look_back, :] = ZMI_Predict
        plt.plot(ZMI_PredictPlot)
        plt.savefig('ZMI_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################桃園市######################################
        dataframe_TYN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\桃園市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TYN=dataframe_TYN.drop(dataframe_TYN[dataframe_TYN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TYN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TYN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TYN_Predict = scaler.inverse_transform(TYN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TYN_PredictPlot = numpy.empty_like(dataset)
        TYN_PredictPlot[:, :] = numpy.nan
        TYN_PredictPlot[look_back:len(TYN_Predict)+look_back, :] = TYN_Predict
        plt.plot(TYN_PredictPlot)
        plt.savefig('TYN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台中市######################################
        model_TXG = load_model('models/taichung_rnn+lstm_1000_1.h5')
        dataframe_TXG = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台中市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TXG=dataframe_TXG.drop(dataframe_TXG[dataframe_TXG['確定病例數']==0].index,axis=0)
        dataset = dataframe_TXG.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TXG_Predict = model_TXG.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TXG_Predict = scaler.inverse_transform(TXG_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TXG_PredictPlot = numpy.empty_like(dataset)
        TXG_PredictPlot[:, :] = numpy.nan
        TXG_PredictPlot[look_back:len(TXG_Predict)+look_back, :] = TXG_Predict
        plt.plot(TXG_PredictPlot)
        plt.savefig('TXG_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################彰化縣######################################
        dataframe_CHW = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CHW=dataframe_CHW.drop(dataframe_CHW[dataframe_CHW['確定病例數']==0].index,axis=0)
        dataset = dataframe_CHW.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CHW_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CHW_Predict = scaler.inverse_transform(CHW_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CHW_PredictPlot = numpy.empty_like(dataset)
        CHW_PredictPlot[:, :] = numpy.nan
        CHW_PredictPlot[look_back:len(CHW_Predict)+look_back, :] = CHW_Predict
        plt.plot(CHW_PredictPlot)
        plt.savefig('CHW_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################南投縣######################################
        model_NTC = load_model('models/nantou_rnn+lstm_1000_1.h5')
        dataframe_NTC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTC=dataframe_NTC.drop(dataframe_NTC[dataframe_NTC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTC_Predict = model_NTC.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTC_Predict = scaler.inverse_transform(NTC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTC_PredictPlot = numpy.empty_like(dataset)
        NTC_PredictPlot[:, :] = numpy.nan
        NTC_PredictPlot[look_back:len(NTC_Predict)+look_back, :] = NTC_Predict
        plt.plot(NTC_PredictPlot)
        plt.savefig('NTC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################雲林縣######################################
        dataframe_YUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\雲林縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_YUN=dataframe_YUN.drop(dataframe_YUN[dataframe_YUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_YUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        YUN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        YUN_Predict = scaler.inverse_transform(YUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        YUN_PredictPlot = numpy.empty_like(dataset)
        YUN_PredictPlot[:, :] = numpy.nan
        YUN_PredictPlot[look_back:len(YUN_Predict)+look_back, :] = YUN_Predict
        plt.plot(YUN_PredictPlot)
        plt.savefig('YUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義縣######################################
        dataframe_CYI0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI0=dataframe_CYI0.drop(dataframe_CYI0[dataframe_CYI0['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI0_Predict = scaler.inverse_transform(CYI0_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI0_PredictPlot = numpy.empty_like(dataset)
        CYI0_PredictPlot[:, :] = numpy.nan
        CYI0_PredictPlot[look_back:len(CYI0_Predict)+look_back, :] = CYI0_Predict
        plt.plot(CYI0_PredictPlot)
        plt.savefig('CYI0_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義市######################################
        dataframe_CYI1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI1=dataframe_CYI1.drop(dataframe_CYI1[dataframe_CYI1['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI1_Predict = scaler.inverse_transform(CYI1_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI1_PredictPlot = numpy.empty_like(dataset)
        CYI1_PredictPlot[:, :] = numpy.nan
        CYI1_PredictPlot[look_back:len(CYI1_Predict)+look_back, :] = CYI1_Predict
        plt.plot(CYI1_PredictPlot)
        plt.savefig('CYI1_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台南市######################################
        model_TNN = load_model('models/tainan_rnn+lstm_1000_1.h5')
        dataframe_TNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台南市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TNN=dataframe_TNN.drop(dataframe_TNN[dataframe_TNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TNN_Predict = model_TNN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TNN_Predict = scaler.inverse_transform(TNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TNN_PredictPlot = numpy.empty_like(dataset)
        TNN_PredictPlot[:, :] = numpy.nan
        TNN_PredictPlot[look_back:len(TNN_Predict)+look_back, :] = TNN_Predict
        plt.plot(TNN_PredictPlot)
        plt.savefig('TNN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################高雄市######################################
        dataframe_KHH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\高雄市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KHH=dataframe_KHH.drop(dataframe_KHH[dataframe_KHH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KHH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KHH_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KHH_Predict = scaler.inverse_transform(KHH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KHH_PredictPlot = numpy.empty_like(dataset)
        KHH_PredictPlot[:, :] = numpy.nan
        KHH_PredictPlot[look_back:len(KHH_Predict)+look_back, :] = KHH_Predict
        plt.plot(KHH_PredictPlot)
        plt.savefig('KHH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################屏東縣######################################
        model_PIF = load_model('models/pingtung_rnn+lstm_1000_1.h5')
        dataframe_PIF = pandas.read_csv(r'D:\專題\各縣市每日確診資料\屏東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PIF=dataframe_PIF.drop(dataframe_PIF[dataframe_PIF['確定病例數']==0].index,axis=0)
        dataset = dataframe_PIF.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PIF_Predict = model_PIF.predict(testX)
	    # 回復預測資料值為原始數據的規模
        PIF_Predict = scaler.inverse_transform(PIF_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PIF_PredictPlot = numpy.empty_like(dataset)
        PIF_PredictPlot[:, :] = numpy.nan
        PIF_PredictPlot[look_back:len(PIF_Predict)+look_back, :] = PIF_Predict
        plt.plot(PIF_PredictPlot)
        plt.savefig('PIF_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################宜蘭縣######################################
        model_ILA = load_model('models/yilan_rnn+lstm_1000_1.h5')
        dataframe_ILA = pandas.read_csv(r'D:\專題\各縣市每日確診資料\宜蘭縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ILA=dataframe_ILA.drop(dataframe_ILA[dataframe_ILA['確定病例數']==0].index,axis=0)
        dataset = dataframe_ILA.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ILA_Predict = model_ILA.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ILA_Predict = scaler.inverse_transform(ILA_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ILA_PredictPlot = numpy.empty_like(dataset)
        ILA_PredictPlot[:, :] = numpy.nan
        ILA_PredictPlot[look_back:len(ILA_Predict)+look_back, :] = ILA_Predict
        plt.plot(ILA_PredictPlot)
        plt.savefig('ILA_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################花蓮縣######################################
        model_HUN = load_model('models/hualien_rnn+lstm_1000_1.h5')
        dataframe_HUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\花蓮縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HUN=dataframe_HUN.drop(dataframe_HUN[dataframe_HUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_HUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HUN_Predict = model_HUN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HUN_Predict = scaler.inverse_transform(HUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HUN_PredictPlot = numpy.empty_like(dataset)
        HUN_PredictPlot[:, :] = numpy.nan
        HUN_PredictPlot[look_back:len(HUN_Predict)+look_back, :] = HUN_Predict
        plt.plot(HUN_PredictPlot)
        plt.savefig('HUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台東縣######################################
        model_TTT = load_model('models/taitung_rnn+lstm_1000_1.h5')
        dataframe_TTT = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TTT=dataframe_TTT.drop(dataframe_TTT[dataframe_TTT['確定病例數']==0].index,axis=0)
        dataset = dataframe_TTT.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TTT_Predict = model_TTT.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TTT_Predict = scaler.inverse_transform(TTT_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TTT_PredictPlot = numpy.empty_like(dataset)
        TTT_PredictPlot[:, :] = numpy.nan
        TTT_PredictPlot[look_back:len(TTT_Predict)+look_back, :] = TTT_Predict
        plt.plot(TTT_PredictPlot)
        plt.savefig('TTT_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################澎湖縣######################################
        dataframe_PEH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\澎湖縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PEH=dataframe_PEH.drop(dataframe_PEH[dataframe_PEH['確定病例數']==0].index,axis=0)
        dataset = dataframe_PEH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PEH_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        PEH_Predict = scaler.inverse_transform(PEH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PEH_PredictPlot = numpy.empty_like(dataset)
        PEH_PredictPlot[:, :] = numpy.nan
        PEH_PredictPlot[look_back:len(PEH_Predict)+look_back, :] = PEH_Predict
        plt.plot(PEH_PredictPlot)
        plt.savefig('PEH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################金門縣######################################
        model_KNH = load_model('models/kinmen_rnn+lstm_1000_1.h5')
        dataframe_KNH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\金門縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KNH=dataframe_KNH.drop(dataframe_KNH[dataframe_KNH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KNH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KNH_Predict = model_KNH.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KNH_Predict = scaler.inverse_transform(KNH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KNH_PredictPlot = numpy.empty_like(dataset)
        KNH_PredictPlot[:, :] = numpy.nan
        KNH_PredictPlot[look_back:len(KNH_Predict)+look_back, :] = KNH_Predict
        plt.plot(KNH_PredictPlot)
        plt.savefig('KNH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################連江縣######################################
        #model_KNH = load_model('models/lienchiang_rnn+lstm_5000_4.h5')
        dataframe_LNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\連江縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_LNN=dataframe_LNN.drop(dataframe_LNN[dataframe_LNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_LNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 3
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        LNN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        LNN_Predict = scaler.inverse_transform(LNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        LNN_PredictPlot = numpy.empty_like(dataset)
        LNN_PredictPlot[:, :] = numpy.nan
        LNN_PredictPlot[look_back:len(LNN_Predict)+look_back, :] = LNN_Predict
        plt.plot(LNN_PredictPlot)
        plt.savefig('LNN_plot.png') 
        plt.clf()
	    #################################################################################
        '''

    if(user_input == '一月'):
        predict_days=30
        #print(date_all)
        #print(type(date_all))
        #Load the trained model. (Pickle file)
        model = load_model('models/taichung_rnn+lstm_1000_1.h5')
        ######################################台北市######################################
	    # 載入訓練資料
        model_TPE = load_model('models/taipei_rnn+lstm_1000_1.h5')
        dataframe_TPE = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TPE=dataframe_TPE.drop(dataframe_TPE[dataframe_TPE['確定病例數']==0].index,axis=0)
        dataset = dataframe_TPE.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TPE_Predict = model_TPE.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TPE_Predict = scaler.inverse_transform(TPE_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TPE_PredictPlot = numpy.empty_like(dataset)
        TPE_PredictPlot[:, :] = numpy.nan
        TPE_PredictPlot[look_back:len(TPE_Predict)+look_back, :] = TPE_Predict
        plt.plot(TPE_PredictPlot)
        plt.savefig('./static/TPE_plot.png') 
        plt.clf()
        #################################################################################
        ######################################基隆市######################################
        model_KEL = load_model('models/keelung_rnn+lstm_1000_2.h5')
        dataframe_KEL = pandas.read_csv('files_for_training_model\各縣市每日確診資料\基隆市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KEL=dataframe_KEL.drop(dataframe_KEL[dataframe_KEL['確定病例數']==0].index,axis=0)
        dataset = dataframe_KEL.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KEL_Predict = model_KEL.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KEL_Predict = scaler.inverse_transform(KEL_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    #shift train predictions for plotting
        KEL_PredictPlot = numpy.empty_like(dataset)
        KEL_PredictPlot[:, :] = numpy.nan
        KEL_PredictPlot[look_back:len(KEL_Predict)+look_back, :] = KEL_Predict
        plt.plot(KEL_PredictPlot)
        plt.savefig('KEL_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新北市######################################
        dataframe_NTPC = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTPC=dataframe_NTPC.drop(dataframe_NTPC[dataframe_NTPC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTPC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTPC_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTPC_Predict = scaler.inverse_transform(NTPC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTPC_PredictPlot = numpy.empty_like(dataset)
        NTPC_PredictPlot[:, :] = numpy.nan
        NTPC_PredictPlot[look_back:len(NTPC_Predict)+look_back, :] = NTPC_Predict
        plt.plot(NTPC_PredictPlot)
        plt.savefig('./static/NTPC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新竹市######################################
        dataframe_HSZ1 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新竹市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ1=dataframe_HSZ1.drop(dataframe_HSZ1[dataframe_HSZ1['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ1_Predict = scaler.inverse_transform(HSZ1_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ1_PredictPlot = numpy.empty_like(dataset)
        HSZ1_PredictPlot[:, :] = numpy.nan
        HSZ1_PredictPlot[look_back:len(HSZ1_Predict)+look_back, :] = HSZ1_Predict
        plt.plot(HSZ1_PredictPlot)
        plt.savefig('./static/HSZ1_plot.png')
        plt.clf()
	    #################################################################################
        ######################################新竹縣######################################
        dataframe_HSZ0 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新竹縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ0=dataframe_HSZ0.drop(dataframe_HSZ0[dataframe_HSZ0['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ0_Predict = scaler.inverse_transform(HSZ0_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ0_PredictPlot = numpy.empty_like(dataset)
        HSZ0_PredictPlot[:, :] = numpy.nan
        HSZ0_PredictPlot[look_back:len(HSZ0_Predict)+look_back, :] = HSZ0_Predict
        plt.plot(HSZ0_PredictPlot)
        plt.savefig('./static/HSZ0_plot.png')
        plt.clf()
	    #################################################################################
        ######################################苗栗縣######################################
        dataframe_ZMI = pandas.read_csv('files_for_training_model\各縣市每日確診資料\苗栗縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ZMI=dataframe_ZMI.drop(dataframe_ZMI[dataframe_ZMI['確定病例數']==0].index,axis=0)
        dataset = dataframe_ZMI.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ZMI_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ZMI_Predict = scaler.inverse_transform(ZMI_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ZMI_PredictPlot = numpy.empty_like(dataset)
        ZMI_PredictPlot[:, :] = numpy.nan
        ZMI_PredictPlot[look_back:len(ZMI_Predict)+look_back, :] = ZMI_Predict
        plt.plot(ZMI_PredictPlot)
        plt.savefig('./static/ZMI_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################桃園市######################################
        dataframe_TYN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\桃園市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TYN=dataframe_TYN.drop(dataframe_TYN[dataframe_TYN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TYN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TYN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TYN_Predict = scaler.inverse_transform(TYN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TYN_PredictPlot = numpy.empty_like(dataset)
        TYN_PredictPlot[:, :] = numpy.nan
        TYN_PredictPlot[look_back:len(TYN_Predict)+look_back, :] = TYN_Predict
        plt.plot(TYN_PredictPlot)
        plt.savefig('./static/TYN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台中市######################################
        model_TXG = load_model('models/taichung_rnn+lstm_1000_1.h5')
        dataframe_TXG = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台中市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TXG=dataframe_TXG.drop(dataframe_TXG[dataframe_TXG['確定病例數']==0].index,axis=0)
        dataset = dataframe_TXG.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TXG_Predict = model_TXG.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TXG_Predict = scaler.inverse_transform(TXG_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TXG_PredictPlot = numpy.empty_like(dataset)
        TXG_PredictPlot[:, :] = numpy.nan
        TXG_PredictPlot[look_back:len(TXG_Predict)+look_back, :] = TXG_Predict
        plt.plot(TXG_PredictPlot)
        plt.savefig('./static/TXG_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################彰化縣######################################
        #model_CHW = load_model('models/taichung_rnn+lstm_1000_1.h5')
        dataframe_CHW = pandas.read_csv('files_for_training_model\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CHW=dataframe_CHW.drop(dataframe_CHW[dataframe_CHW['確定病例數']==0].index,axis=0)
        dataset = dataframe_CHW.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CHW_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CHW_Predict = scaler.inverse_transform(CHW_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CHW_PredictPlot = numpy.empty_like(dataset)
        CHW_PredictPlot[:, :] = numpy.nan
        CHW_PredictPlot[look_back:len(CHW_Predict)+look_back, :] = CHW_Predict
        plt.plot(CHW_PredictPlot)
        plt.savefig('./static/CHW_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################南投縣######################################
        model_NTC = load_model('models/nantou_rnn+lstm_1000_1.h5')
        dataframe_NTC = pandas.read_csv('files_for_training_model\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTC=dataframe_NTC.drop(dataframe_NTC[dataframe_NTC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTC_Predict = model_NTC.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTC_Predict = scaler.inverse_transform(NTC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTC_PredictPlot = numpy.empty_like(dataset)
        NTC_PredictPlot[:, :] = numpy.nan
        NTC_PredictPlot[look_back:len(NTC_Predict)+look_back, :] = NTC_Predict
        plt.plot(NTC_PredictPlot)
        plt.savefig('./static/NTC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################雲林縣######################################
        dataframe_YUN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\雲林縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_YUN=dataframe_YUN.drop(dataframe_YUN[dataframe_YUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_YUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        YUN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        YUN_Predict = scaler.inverse_transform(YUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        YUN_PredictPlot = numpy.empty_like(dataset)
        YUN_PredictPlot[:, :] = numpy.nan
        YUN_PredictPlot[look_back:len(YUN_Predict)+look_back, :] = YUN_Predict
        plt.plot(YUN_PredictPlot)
        plt.savefig('./static/YUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義縣######################################
        dataframe_CYI0 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\嘉義縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI0=dataframe_CYI0.drop(dataframe_CYI0[dataframe_CYI0['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI0_Predict = scaler.inverse_transform(CYI0_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI0_PredictPlot = numpy.empty_like(dataset)
        CYI0_PredictPlot[:, :] = numpy.nan
        CYI0_PredictPlot[look_back:len(CYI0_Predict)+look_back, :] = CYI0_Predict
        plt.plot(CYI0_PredictPlot)
        plt.savefig('./static/CYI0_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義市######################################
        dataframe_CYI1 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\嘉義市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI1=dataframe_CYI1.drop(dataframe_CYI1[dataframe_CYI1['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI1_Predict = scaler.inverse_transform(CYI1_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI1_PredictPlot = numpy.empty_like(dataset)
        CYI1_PredictPlot[:, :] = numpy.nan
        CYI1_PredictPlot[look_back:len(CYI1_Predict)+look_back, :] = CYI1_Predict
        plt.plot(CYI1_PredictPlot)
        plt.savefig('./static/CYI1_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台南市######################################
        model_TNN = load_model('models/tainan_rnn+lstm_1000_1.h5')
        dataframe_TNN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台南市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TNN=dataframe_TNN.drop(dataframe_TNN[dataframe_TNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TNN_Predict = model_TNN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TNN_Predict = scaler.inverse_transform(TNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TNN_PredictPlot = numpy.empty_like(dataset)
        TNN_PredictPlot[:, :] = numpy.nan
        TNN_PredictPlot[look_back:len(TNN_Predict)+look_back, :] = TNN_Predict
        plt.plot(TNN_PredictPlot)
        plt.savefig('./static/TNN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################高雄市######################################
        dataframe_KHH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\高雄市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KHH=dataframe_KHH.drop(dataframe_KHH[dataframe_KHH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KHH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KHH_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KHH_Predict = scaler.inverse_transform(KHH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KHH_PredictPlot = numpy.empty_like(dataset)
        KHH_PredictPlot[:, :] = numpy.nan
        KHH_PredictPlot[look_back:len(KHH_Predict)+look_back, :] = KHH_Predict
        plt.plot(KHH_PredictPlot)
        plt.savefig('./static/KHH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################屏東縣######################################
        model_PIF = load_model('models/pingtung_rnn+lstm_1000_1.h5')
        dataframe_PIF = pandas.read_csv('files_for_training_model\各縣市每日確診資料\屏東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PIF=dataframe_PIF.drop(dataframe_PIF[dataframe_PIF['確定病例數']==0].index,axis=0)
        dataset = dataframe_PIF.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PIF_Predict = model_PIF.predict(testX)
	    # 回復預測資料值為原始數據的規模
        PIF_Predict = scaler.inverse_transform(PIF_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PIF_PredictPlot = numpy.empty_like(dataset)
        PIF_PredictPlot[:, :] = numpy.nan
        PIF_PredictPlot[look_back:len(PIF_Predict)+look_back, :] = PIF_Predict
        plt.plot(PIF_PredictPlot)
        plt.savefig('./static/PIF_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################宜蘭縣######################################
        model_ILA = load_model('models/yilan_rnn+lstm_1000_1.h5')
        dataframe_ILA = pandas.read_csv('files_for_training_model\各縣市每日確診資料\宜蘭縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ILA=dataframe_ILA.drop(dataframe_ILA[dataframe_ILA['確定病例數']==0].index,axis=0)
        dataset = dataframe_ILA.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ILA_Predict = model_ILA.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ILA_Predict = scaler.inverse_transform(ILA_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ILA_PredictPlot = numpy.empty_like(dataset)
        ILA_PredictPlot[:, :] = numpy.nan
        ILA_PredictPlot[look_back:len(ILA_Predict)+look_back, :] = ILA_Predict
        plt.plot(ILA_PredictPlot)
        plt.savefig('./static/ILA_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################花蓮縣######################################
        model_HUN = load_model('models/hualien_rnn+lstm_1000_1.h5')
        dataframe_HUN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\花蓮縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HUN=dataframe_HUN.drop(dataframe_HUN[dataframe_HUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_HUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HUN_Predict = model_HUN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HUN_Predict = scaler.inverse_transform(HUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HUN_PredictPlot = numpy.empty_like(dataset)
        HUN_PredictPlot[:, :] = numpy.nan
        HUN_PredictPlot[look_back:len(HUN_Predict)+look_back, :] = HUN_Predict
        plt.plot(HUN_PredictPlot)
        plt.savefig('./static/HUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台東縣######################################
        model_TTT = load_model('models/taitung_rnn+lstm_1000_1.h5')
        dataframe_TTT = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TTT=dataframe_TTT.drop(dataframe_TTT[dataframe_TTT['確定病例數']==0].index,axis=0)
        dataset = dataframe_TTT.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TTT_Predict = model_TTT.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TTT_Predict = scaler.inverse_transform(TTT_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TTT_PredictPlot = numpy.empty_like(dataset)
        TTT_PredictPlot[:, :] = numpy.nan
        TTT_PredictPlot[look_back:len(TTT_Predict)+look_back, :] = TTT_Predict
        plt.plot(TTT_PredictPlot)
        plt.savefig('./static/TTT_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################澎湖縣######################################
        model_PEH = load_model('models/penghu_rnn+lstm_5000_2.h5')
        dataframe_PEH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\澎湖縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PEH=dataframe_PEH.drop(dataframe_PEH[dataframe_PEH['確定病例數']==0].index,axis=0)
        dataset = dataframe_PEH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PEH_Predict = model_PEH.predict(testX)
	    # 回復預測資料值為原始數據的規模
        PEH_Predict = scaler.inverse_transform(PEH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PEH_PredictPlot = numpy.empty_like(dataset)
        PEH_PredictPlot[:, :] = numpy.nan
        PEH_PredictPlot[look_back:len(PEH_Predict)+look_back, :] = PEH_Predict
        plt.plot(PEH_PredictPlot)
        plt.savefig('./static/PEH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################金門縣######################################
        model_KNH = load_model('models/kinmen_rnn+lstm_1000_1.h5')
        dataframe_KNH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\金門縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KNH=dataframe_KNH.drop(dataframe_KNH[dataframe_KNH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KNH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KNH_Predict = model_KNH.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KNH_Predict = scaler.inverse_transform(KNH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KNH_PredictPlot = numpy.empty_like(dataset)
        KNH_PredictPlot[:, :] = numpy.nan
        KNH_PredictPlot[look_back:len(KNH_Predict)+look_back, :] = KNH_Predict
        plt.plot(KNH_PredictPlot)
        plt.savefig('./static/KNH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################連江縣######################################
        model_LNN = load_model('models/lienchiang_rnn+lstm_1000_2.h5')
        dataframe_LNN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\連江縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_LNN=dataframe_LNN.drop(dataframe_LNN[dataframe_LNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_LNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 30
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        LNN_Predict = model_LNN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        LNN_Predict = scaler.inverse_transform(LNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        LNN_PredictPlot = numpy.empty_like(dataset)
        LNN_PredictPlot[:, :] = numpy.nan
        LNN_PredictPlot[look_back:len(LNN_Predict)+look_back, :] = LNN_Predict
        plt.plot(LNN_PredictPlot)
        plt.savefig('./static/LNN_plot.png') 
        plt.clf()
    if(user_input == '一周'):
        predict_days=7
        #print(date_all)
        #print(type(date_all))
        #Load the trained model. (Pickle file)
        model = load_model('models/taichung_rnn+lstm_1000_1.h5')
        ######################################台北市######################################
	    # 載入訓練資料
        model_TPE = load_model('models/taipei_rnn+lstm_1000_1.h5')
        dataframe_TPE = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TPE=dataframe_TPE.drop(dataframe_TPE[dataframe_TPE['確定病例數']==0].index,axis=0)
        dataset = dataframe_TPE.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TPE_Predict = model_TPE.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TPE_Predict = scaler.inverse_transform(TPE_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TPE_PredictPlot = numpy.empty_like(dataset)
        TPE_PredictPlot[:, :] = numpy.nan
        TPE_PredictPlot[look_back:len(TPE_Predict)+look_back, :] = TPE_Predict
        plt.plot(TPE_PredictPlot)
        plt.savefig('./static/TPE_plot.png') 
        plt.clf()
        #################################################################################
        ######################################基隆市######################################
        model_KEL = load_model('models/keelung_rnn+lstm_1000_2.h5')
        dataframe_KEL = pandas.read_csv('files_for_training_model\各縣市每日確診資料\基隆市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KEL=dataframe_KEL.drop(dataframe_KEL[dataframe_KEL['確定病例數']==0].index,axis=0)
        dataset = dataframe_KEL.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KEL_Predict = model_KEL.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KEL_Predict = scaler.inverse_transform(KEL_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    #shift train predictions for plotting
        KEL_PredictPlot = numpy.empty_like(dataset)
        KEL_PredictPlot[:, :] = numpy.nan
        KEL_PredictPlot[look_back:len(KEL_Predict)+look_back, :] = KEL_Predict
        plt.plot(KEL_PredictPlot)
        plt.savefig('./static/KEL_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新北市######################################
        dataframe_NTPC = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTPC=dataframe_NTPC.drop(dataframe_NTPC[dataframe_NTPC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTPC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTPC_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTPC_Predict = scaler.inverse_transform(NTPC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTPC_PredictPlot = numpy.empty_like(dataset)
        NTPC_PredictPlot[:, :] = numpy.nan
        NTPC_PredictPlot[look_back:len(NTPC_Predict)+look_back, :] = NTPC_Predict
        plt.plot(NTPC_PredictPlot)
        plt.savefig('./static/NTPC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################新竹市######################################
        dataframe_HSZ1 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新竹市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ1=dataframe_HSZ1.drop(dataframe_HSZ1[dataframe_HSZ1['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ1_Predict = scaler.inverse_transform(HSZ1_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ1_PredictPlot = numpy.empty_like(dataset)
        HSZ1_PredictPlot[:, :] = numpy.nan
        HSZ1_PredictPlot[look_back:len(HSZ1_Predict)+look_back, :] = HSZ1_Predict
        plt.plot(HSZ1_PredictPlot)
        plt.savefig('./static/HSZ1_plot.png')
        plt.clf()
	    #################################################################################
        ######################################新竹縣######################################
        dataframe_HSZ0 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\新竹縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HSZ0=dataframe_HSZ0.drop(dataframe_HSZ0[dataframe_HSZ0['確定病例數']==0].index,axis=0)
        dataset = dataframe_HSZ0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HSZ0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HSZ0_Predict = scaler.inverse_transform(HSZ0_Predict)
        testY = scaler.inverse_transform([testY])
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HSZ0_PredictPlot = numpy.empty_like(dataset)
        HSZ0_PredictPlot[:, :] = numpy.nan
        HSZ0_PredictPlot[look_back:len(HSZ0_Predict)+look_back, :] = HSZ0_Predict
        plt.plot(HSZ0_PredictPlot)
        plt.savefig('./static/HSZ0_plot.png')
        plt.clf()
	    #################################################################################
        ######################################苗栗縣######################################
        dataframe_ZMI = pandas.read_csv('files_for_training_model\各縣市每日確診資料\苗栗縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ZMI=dataframe_ZMI.drop(dataframe_ZMI[dataframe_ZMI['確定病例數']==0].index,axis=0)
        dataset = dataframe_ZMI.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ZMI_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ZMI_Predict = scaler.inverse_transform(ZMI_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ZMI_PredictPlot = numpy.empty_like(dataset)
        ZMI_PredictPlot[:, :] = numpy.nan
        ZMI_PredictPlot[look_back:len(ZMI_Predict)+look_back, :] = ZMI_Predict
        plt.plot(ZMI_PredictPlot)
        plt.savefig('./static/ZMI_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################桃園市######################################
        dataframe_TYN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\桃園市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TYN=dataframe_TYN.drop(dataframe_TYN[dataframe_TYN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TYN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TYN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TYN_Predict = scaler.inverse_transform(TYN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TYN_PredictPlot = numpy.empty_like(dataset)
        TYN_PredictPlot[:, :] = numpy.nan
        TYN_PredictPlot[look_back:len(TYN_Predict)+look_back, :] = TYN_Predict
        plt.plot(TYN_PredictPlot)
        plt.savefig('./static/TYN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台中市######################################
        model_TXG = load_model('models/taichung_rnn+lstm_1000_1.h5')
        dataframe_TXG = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台中市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TXG=dataframe_TXG.drop(dataframe_TXG[dataframe_TXG['確定病例數']==0].index,axis=0)
        dataset = dataframe_TXG.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TXG_Predict = model_TXG.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TXG_Predict = scaler.inverse_transform(TXG_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TXG_PredictPlot = numpy.empty_like(dataset)
        TXG_PredictPlot[:, :] = numpy.nan
        TXG_PredictPlot[look_back:len(TXG_Predict)+look_back, :] = TXG_Predict
        plt.plot(TXG_PredictPlot)
        plt.savefig('./static/TXG_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################彰化縣######################################
        dataframe_CHW = pandas.read_csv('files_for_training_model\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CHW=dataframe_CHW.drop(dataframe_CHW[dataframe_CHW['確定病例數']==0].index,axis=0)
        dataset = dataframe_CHW.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CHW_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CHW_Predict = scaler.inverse_transform(CHW_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CHW_PredictPlot = numpy.empty_like(dataset)
        CHW_PredictPlot[:, :] = numpy.nan
        CHW_PredictPlot[look_back:len(CHW_Predict)+look_back, :] = CHW_Predict
        plt.plot(CHW_PredictPlot)
        plt.savefig('./static/CHW_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################南投縣######################################
        model_NTC = load_model('models/nantou_rnn+lstm_1000_1.h5')
        dataframe_NTC = pandas.read_csv('files_for_training_model\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_NTC=dataframe_NTC.drop(dataframe_NTC[dataframe_NTC['確定病例數']==0].index,axis=0)
        dataset = dataframe_NTC.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        NTC_Predict = model_NTC.predict(testX)
	    # 回復預測資料值為原始數據的規模
        NTC_Predict = scaler.inverse_transform(NTC_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        NTC_PredictPlot = numpy.empty_like(dataset)
        NTC_PredictPlot[:, :] = numpy.nan
        NTC_PredictPlot[look_back:len(NTC_Predict)+look_back, :] = NTC_Predict
        plt.plot(NTC_PredictPlot)
        plt.savefig('./static/NTC_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################雲林縣######################################
        dataframe_YUN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\雲林縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_YUN=dataframe_YUN.drop(dataframe_YUN[dataframe_YUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_YUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        YUN_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        YUN_Predict = scaler.inverse_transform(YUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        YUN_PredictPlot = numpy.empty_like(dataset)
        YUN_PredictPlot[:, :] = numpy.nan
        YUN_PredictPlot[look_back:len(YUN_Predict)+look_back, :] = YUN_Predict
        plt.plot(YUN_PredictPlot)
        plt.savefig('./static/YUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義縣######################################
        dataframe_CYI0 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\嘉義縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI0=dataframe_CYI0.drop(dataframe_CYI0[dataframe_CYI0['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI0.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI0_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI0_Predict = scaler.inverse_transform(CYI0_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI0_PredictPlot = numpy.empty_like(dataset)
        CYI0_PredictPlot[:, :] = numpy.nan
        CYI0_PredictPlot[look_back:len(CYI0_Predict)+look_back, :] = CYI0_Predict
        plt.plot(CYI0_PredictPlot)
        plt.savefig('./static/CYI0_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################嘉義市######################################
        dataframe_CYI1 = pandas.read_csv('files_for_training_model\各縣市每日確診資料\嘉義市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_CYI1=dataframe_CYI1.drop(dataframe_CYI1[dataframe_CYI1['確定病例數']==0].index,axis=0)
        dataset = dataframe_CYI1.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        CYI1_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        CYI1_Predict = scaler.inverse_transform(CYI1_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        CYI1_PredictPlot = numpy.empty_like(dataset)
        CYI1_PredictPlot[:, :] = numpy.nan
        CYI1_PredictPlot[look_back:len(CYI1_Predict)+look_back, :] = CYI1_Predict
        plt.plot(CYI1_PredictPlot)
        plt.savefig('./static/CYI1_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台南市######################################
        model_TNN = load_model('models/tainan_rnn+lstm_1000_1.h5')
        dataframe_TNN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台南市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TNN=dataframe_TNN.drop(dataframe_TNN[dataframe_TNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_TNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TNN_Predict = model_TNN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TNN_Predict = scaler.inverse_transform(TNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TNN_PredictPlot = numpy.empty_like(dataset)
        TNN_PredictPlot[:, :] = numpy.nan
        TNN_PredictPlot[look_back:len(TNN_Predict)+look_back, :] = TNN_Predict
        plt.plot(TNN_PredictPlot)
        plt.savefig('./static/TNN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################高雄市######################################
        dataframe_KHH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\高雄市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KHH=dataframe_KHH.drop(dataframe_KHH[dataframe_KHH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KHH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KHH_Predict = model.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KHH_Predict = scaler.inverse_transform(KHH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KHH_PredictPlot = numpy.empty_like(dataset)
        KHH_PredictPlot[:, :] = numpy.nan
        KHH_PredictPlot[look_back:len(KHH_Predict)+look_back, :] = KHH_Predict
        plt.plot(KHH_PredictPlot)
        plt.savefig('./static/KHH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################屏東縣######################################
        model_PIF = load_model('models/pingtung_rnn+lstm_1000_1.h5')
        dataframe_PIF = pandas.read_csv('files_for_training_model\各縣市每日確診資料\屏東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PIF=dataframe_PIF.drop(dataframe_PIF[dataframe_PIF['確定病例數']==0].index,axis=0)
        dataset = dataframe_PIF.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PIF_Predict = model_PIF.predict(testX)
	    # 回復預測資料值為原始數據的規模
        PIF_Predict = scaler.inverse_transform(PIF_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PIF_PredictPlot = numpy.empty_like(dataset)
        PIF_PredictPlot[:, :] = numpy.nan
        PIF_PredictPlot[look_back:len(PIF_Predict)+look_back, :] = PIF_Predict
        plt.plot(PIF_PredictPlot)
        plt.savefig('./static/PIF_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################宜蘭縣######################################
        model_ILA = load_model('models/yilan_rnn+lstm_1000_1.h5')
        dataframe_ILA = pandas.read_csv('files_for_training_model\各縣市每日確診資料\宜蘭縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_ILA=dataframe_ILA.drop(dataframe_ILA[dataframe_ILA['確定病例數']==0].index,axis=0)
        dataset = dataframe_ILA.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        ILA_Predict = model_ILA.predict(testX)
	    # 回復預測資料值為原始數據的規模
        ILA_Predict = scaler.inverse_transform(ILA_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        ILA_PredictPlot = numpy.empty_like(dataset)
        ILA_PredictPlot[:, :] = numpy.nan
        ILA_PredictPlot[look_back:len(ILA_Predict)+look_back, :] = ILA_Predict
        plt.plot(ILA_PredictPlot)
        plt.savefig('./static/ILA_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################花蓮縣######################################
        model_HUN = load_model('models/hualien_rnn+lstm_1000_1.h5')
        dataframe_HUN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\花蓮縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_HUN=dataframe_HUN.drop(dataframe_HUN[dataframe_HUN['確定病例數']==0].index,axis=0)
        dataset = dataframe_HUN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        HUN_Predict = model_HUN.predict(testX)
	    # 回復預測資料值為原始數據的規模
        HUN_Predict = scaler.inverse_transform(HUN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        HUN_PredictPlot = numpy.empty_like(dataset)
        HUN_PredictPlot[:, :] = numpy.nan
        HUN_PredictPlot[look_back:len(HUN_Predict)+look_back, :] = HUN_Predict
        plt.plot(HUN_PredictPlot)
        plt.savefig('./static/HUN_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################台東縣######################################
        model_TTT = load_model('models/taitung_rnn+lstm_1000_1.h5')
        dataframe_TTT = pandas.read_csv('files_for_training_model\各縣市每日確診資料\台東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_TTT=dataframe_TTT.drop(dataframe_TTT[dataframe_TTT['確定病例數']==0].index,axis=0)
        dataset = dataframe_TTT.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        TTT_Predict = model_TTT.predict(testX)
	    # 回復預測資料值為原始數據的規模
        TTT_Predict = scaler.inverse_transform(TTT_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        TTT_PredictPlot = numpy.empty_like(dataset)
        TTT_PredictPlot[:, :] = numpy.nan
        TTT_PredictPlot[look_back:len(TTT_Predict)+look_back, :] = TTT_Predict
        plt.plot(TTT_PredictPlot)
        plt.savefig('./static/TTT_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################澎湖縣######################################
        model_PEH = load_model('models/penghu_rnn+lstm_5000_2.h5')
        dataframe_PEH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\澎湖縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_PEH=dataframe_PEH.drop(dataframe_PEH[dataframe_PEH['確定病例數']==0].index,axis=0)
        dataset = dataframe_PEH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        PEH_Predict = model_PEH.predict(testX)

        PEH_Predict=PEH_Predict.reshape(-1, 1)

	    # 回復預測資料值為原始數據的規模
        PEH_Predict = scaler.inverse_transform(PEH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        PEH_PredictPlot = numpy.empty_like(dataset)
        PEH_PredictPlot[:, :] = numpy.nan
        PEH_PredictPlot[look_back:len(PEH_Predict)+look_back, :] = PEH_Predict
        plt.plot(PEH_PredictPlot)
        plt.savefig('./static/PEH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################金門縣######################################
        model_KNH = load_model('models/kinmen_rnn+lstm_1000_1.h5')
        dataframe_KNH = pandas.read_csv('files_for_training_model\各縣市每日確診資料\金門縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_KNH=dataframe_KNH.drop(dataframe_KNH[dataframe_KNH['確定病例數']==0].index,axis=0)
        dataset = dataframe_KNH.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        KNH_Predict = model_KNH.predict(testX)
	    # 回復預測資料值為原始數據的規模
        KNH_Predict = scaler.inverse_transform(KNH_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        KNH_PredictPlot = numpy.empty_like(dataset)
        KNH_PredictPlot[:, :] = numpy.nan
        KNH_PredictPlot[look_back:len(KNH_Predict)+look_back, :] = KNH_Predict
        plt.plot(KNH_PredictPlot)
        plt.savefig('./static/KNH_plot.png') 
        plt.clf()
	    #################################################################################
        ######################################連江縣######################################
        model_LNN = load_model('models/lienchiang_rnn+lstm_1000_2.h5')
        dataframe_LNN = pandas.read_csv('files_for_training_model\各縣市每日確診資料\連江縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
        dataframe_LNN=dataframe_LNN.drop(dataframe_LNN[dataframe_LNN['確定病例數']==0].index,axis=0)
        dataset = dataframe_LNN.values
	    # 正規化(normalize) 資料，使資料值介於[0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        tf.random.set_seed(7)
        test_size = 9
        test = dataset[len(dataset)-test_size:,:]
	    # 產生 (X, Y) 資料集, Y 是下一期的確診數(reshape into X=t and Y=t+1)
        look_back = 1
        testX, testY = create_dataset(test, look_back)
	    # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        LNN_Predict = model_LNN.predict(testX)

        LNN_Predict=LNN_Predict.reshape(-1, 1)
	    # 回復預測資料值為原始數據的規模
        LNN_Predict = scaler.inverse_transform(LNN_Predict)
        testY = scaler.inverse_transform([testY]) 
	    # 畫訓練資料趨勢圖
	    # shift train predictions for plotting
        LNN_PredictPlot = numpy.empty_like(dataset)
        LNN_PredictPlot[:, :] = numpy.nan
        LNN_PredictPlot[look_back:len(LNN_Predict)+look_back, :] = LNN_Predict
        plt.plot(LNN_PredictPlot)
        plt.savefig('./static/LNN_plot.png') 
        plt.clf()
	    #################################################################################
        KEL_Predict[KEL_Predict < 0] = 0
        NTPC_Predict[NTPC_Predict < 0] = 0
        TPE_Predict[TPE_Predict < 0] = 0
        TYN_Predict[TYN_Predict < 0] = 0
        HSZ0_Predict[HSZ0_Predict < 0] = 0
        HSZ1_Predict[HSZ1_Predict < 0] = 0
        ZMI_Predict[ZMI_Predict < 0] = 0
        TXG_Predict[TXG_Predict < 0] = 0
        CHW_Predict[CHW_Predict < 0] = 0
        NTC_Predict[NTC_Predict < 0] = 0
        YUN_Predict[YUN_Predict < 0] = 0
        CYI0_Predict[CYI0_Predict < 0] = 0
        CYI1_Predict[CYI1_Predict < 0] = 0
        TNN_Predict[TNN_Predict < 0] = 0
        KHH_Predict[KHH_Predict < 0] = 0
        PIF_Predict[PIF_Predict < 0] = 0
        ILA_Predict[ILA_Predict < 0] = 0
        HUN_Predict[HUN_Predict < 0] = 0
        TTT_Predict[TTT_Predict < 0] = 0
        PEH_Predict[PEH_Predict < 0] = 0
        KNH_Predict[KNH_Predict < 0] = 0
        LNN_Predict[LNN_Predict < 0] = 0

    data_re01 = [int(x) for x in KEL_Predict]
    data_re02 = [int(x) for x in NTPC_Predict]
    data_re03 = [int(x) for x in TPE_Predict]
    data_re04 = [int(x) for x in TYN_Predict]
    data_re05 = [int(x) for x in HSZ0_Predict]
    data_re06 = [int(x) for x in HSZ1_Predict]
    data_re07 = [int(x) for x in ZMI_Predict]
    data_re08 = [int(x) for x in TXG_Predict]
    data_re09 = [int(x) for x in CHW_Predict]
    data_re10 = [int(x) for x in NTC_Predict]
    data_re11 = [int(x) for x in YUN_Predict]
    data_re12 = [int(x) for x in CYI0_Predict]
    data_re13 = [int(x) for x in CYI1_Predict]
    data_re14 = [int(x) for x in TNN_Predict]
    data_re15 = [int(x) for x in KHH_Predict]
    data_re16 = [int(x) for x in PIF_Predict]
    data_re17 = [int(x) for x in ILA_Predict]
    data_re18 = [int(x) for x in HUN_Predict]
    data_re19 = [int(x) for x in TTT_Predict]
    data_re20 = [int(x) for x in PEH_Predict]
    data_re21 = [int(x) for x in KNH_Predict]
    data_re22 = [int(x) for x in LNN_Predict]


    # 全國人數加總list
    data_all = []
    for i in range(predict_days): 
        total = (int(data_re01[i]) + int(data_re02[i]) + int(data_re03[i]) + int(data_re04[i]) + 
                 int(data_re05[i]) + int(data_re06[i]) + int(data_re07[i]) + int(data_re08[i]) + 
                 int(data_re09[i]) + int(data_re10[i]) + int(data_re11[i]) + int(data_re12[i]) + 
                 int(data_re13[i]) + int(data_re14[i]) + int(data_re15[i]) + int(data_re16[i]) + 
                 int(data_re17[i]) + int(data_re18[i]) + int(data_re19[i]) + int(data_re20[i]) + 
                 int(data_re21[i]) + int(data_re22[i]))
        #print(total)
        data_all.append(total)
    #print(data_all)
    # 日期最後一天的確診人數
    today_data = data_all[-1]
    data_KEL = data_re01[-1]
    data_NTPC = data_re02[-1]
    data_TPE = data_re03[-1]
    data_TYN = data_re04[-1]
    data_HSZ0 = data_re05[-1]
    data_HSZ1 = data_re06[-1]
    data_ZMI = data_re07[-1]
    data_TXG = data_re08[-1]
    data_CHW = data_re09[-1]
    data_NTC = data_re10[-1]
    data_YUN = data_re11[-1]
    data_CYI0 = data_re12[-1]
    data_CYI1 = data_re13[-1]
    data_TNN = data_re14[-1]
    data_KHH = data_re15[-1]
    data_PIF = data_re16[-1]
    data_ILA = data_re17[-1]
    data_HUN = data_re18[-1]
    data_TTT = data_re19[-1]
    data_PEH = data_re20[-1]
    data_KNH = data_re21[-1]
    data_LNN = data_re22[-1]
    
        
    # 合併list，全部日期 + 全國確診人數
    #list_all = dict(zip(date_all, data_all))

    return render_template("third_web.html", **locals())

if __name__ == '__main__':
    print('####  Flask Start... ####')
    #app.debug = True
    #app.use_reloader=False
    app.run(host='0.0.0.0')
    #app.run()
    #db.close() #關閉連線


# In[ ]:




