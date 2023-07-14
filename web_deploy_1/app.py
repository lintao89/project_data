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
import re, datetime
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""


import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

plt.switch_backend('agg')

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('first_web.html')


# 產生 (X, Y) 資料集, Y 是下一期的感染人數
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = request.values['user_input']
    if(user_input == '一月'):
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
        dataframe_KEL = pandas.read_csv(r'D:\專題\各縣市每日確診資料\基隆市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_NTPC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HSZ1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HSZ0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_ZMI = pandas.read_csv(r'D:\專題\各縣市每日確診資料\苗栗縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TYN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\桃園市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TXG = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台中市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CHW = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_NTC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_YUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\雲林縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CYI0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CYI1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台南市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_KHH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\高雄市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_PIF = pandas.read_csv(r'D:\專題\各縣市每日確診資料\屏東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_ILA = pandas.read_csv(r'D:\專題\各縣市每日確診資料\宜蘭縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\花蓮縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TTT = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_PEH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\澎湖縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_KNH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\金門縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_LNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\連江縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_KEL = pandas.read_csv(r'D:\專題\各縣市每日確診資料\基隆市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_NTPC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新北市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HSZ1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HSZ0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\新竹縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_ZMI = pandas.read_csv(r'D:\專題\各縣市每日確診資料\苗栗縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TYN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\桃園市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TXG = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台中市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CHW = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_NTC = pandas.read_csv(r'D:\專題\各縣市每日確診資料\彰化縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_YUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\雲林縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CYI0 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_CYI1 = pandas.read_csv(r'D:\專題\各縣市每日確診資料\嘉義市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台南市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_KHH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\高雄市每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_PIF = pandas.read_csv(r'D:\專題\各縣市每日確診資料\屏東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_ILA = pandas.read_csv(r'D:\專題\各縣市每日確診資料\宜蘭縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_HUN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\花蓮縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_TTT = pandas.read_csv(r'D:\專題\各縣市每日確診資料\台東縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_PEH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\澎湖縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_KNH = pandas.read_csv(r'D:\專題\各縣市每日確診資料\金門縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
        dataframe_LNN = pandas.read_csv(r'D:\專題\各縣市每日確診資料\連江縣每日確診數.csv', usecols=[2], engine='python', skipfooter=0,encoding='utf-8')
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
    data_re01=[int(x) for x in KEL_Predict]
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
    for i in range(7): 
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
	#return render_template('first_web.html', prediction_text='Covid population in next week is {}'.format(testPredict))
#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.debug = True
    app.run()
