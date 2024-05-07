'''
@File  :utils.py.py
@Author:SunNuan
@Date  :2024/2/19 15:25
@Desc  :
'''
import copy

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR,NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import RBFInterpolator
# from smt.surrogate_models.rbf import RBF
import warnings
import joblib
import seaborn as sns
# warnings.filterwarnings("ignore")
# np.set_printoptions(suppress = True)


def readData(data_path, label):
    '''读取excel数据'''
    raw_data = pd.read_csv(data_path)
    # 将原始数据复制一份
    data = copy.deepcopy(raw_data)
    label = data.pop(label)
    return raw_data, data, label

raw_data, data, label = readData('./data/data_0104_.csv')

# 数据集拆分为训练集和测试集
xTrain,xTest,yTrain,yTest = train_test_split(data,label,test_size=0.1,random_state=420)


def createLrModelSklearn():
    '''
    线性回归算法构建模型
    @return:
    '''
    # 搭建模型
    model = LinearRegression()
    # 开始训练
    model.fit(xTrain, yTrain)
    # 利用预测集预测
    yPredTest = model.predict(xTest)
    yPredTrain = model.predict(xTrain)
    # 计算r2值，评价模型，越大越好，最大值为1
    r2Test = r2_score(yTest, yPredTest).round(3)
    r2Train = r2_score(yTrain, yPredTrain).round(3)
    # 计算mae值，评价模型，越小越好
    maeTest = mean_absolute_error(yTest, yPredTest).round(3)
    maeTrain = mean_absolute_error(yTrain, yPredTrain).round(3)
    print(f'modelname:{model},\ntrainR2:{r2Train},\ntestR2:{r2Test}')
    # 将训练好的模型序列化到本地磁盘
    joblib.dump(model, './models/lrModelSklearn.pkl')
    return maeTest, maeTrain, r2Train, r2Test, model, yPredTest

lrMaeTestSklearn, lrMaeTrainSklearn,lrR2TrainSklearn, \
lrR2TestSklearn, lrModelSklearn, lryPredTestSklearn = createLrModelSklearn()


def createSVRModelSklearn():
    '''创建支持向量机回归模型'''
    model = SVR()
    model.fit(xTrain, yTrain)
    yPredTest = model.predict(xTest)
    yPredTrain = model.predict(xTrain)
    r2Test = r2_score(yTest, yPredTest).round(3)
    r2Train = r2_score(yTrain, yPredTrain).round(3)
    maeTest = mean_absolute_error(yTest, yPredTest).round(3)
    maeTrain = mean_absolute_error(yTrain, yPredTrain).round(3)
    print(f'modelname:{model},\ntrainR2:{r2Train},\ntestR2:{r2Test}')
    joblib.dump(model, './models/svrModelSklearn.pkl')
    return maeTest, maeTrain, r2Train, r2Test, model, yPredTest

svrMaeTestSklearn, svrMaeTrainSklearn,svrR2TrainSklearn,\
svrR2TestSklearn, svrModelSklearn, svryPredTestSklearn = createSVRModelSklearn()

# 特征重要性函数编写
def getFeaturesImportance(model, n, flag, colNames):
    '''获取特征重要性排序,model:模型名字，n：过滤阈值，flag:模型名字，colNames，列名字'''
    randomForestFeaturesImportance = dict(zip(colNames, model.feature_importances_))
    sorted(randomForestFeaturesImportance.items(), key=lambda d: d[1], reverse=True)
    randomForestFeaturesImportance = pd.DataFrame.from_dict(randomForestFeaturesImportance,orient='index') \
                                                            .reset_index() \
                                                            .rename(columns={'index':'featureName',0:'score'}) \
                                                            .sort_values(by='score', ascending=False)
    randomForestFeaturesImportance = randomForestFeaturesImportance[randomForestFeaturesImportance['score']>n]
    randomForestFeaturesImportance.to_csv('../data/'+flag+'FeaturesImportance.csv', index=False)
    return randomForestFeaturesImportance

# 随机森林
def createRandomForestModelSklearn():
    '''创建随机森林模型'''
    model = RandomForestRegressor()
    model.fit(xTrain, yTrain)
    yPredTest = model.predict(xTest)
    yPredTrain = model.predict(xTrain)
    r2Test = r2_score(yTest, yPredTest).round(3)
    r2Train = r2_score(yTrain, yPredTrain).round(3)
    maeTest = mean_absolute_error(yTest, yPredTest).round(3)
    maeTrain = mean_absolute_error(yTrain, yPredTrain).round(3)
    print(f'modelname:{model},\ntrainR2:{r2Train},\ntestR2:{r2Test}')
    joblib.dump(model, './models/randomForestModelSklearn.pkl')
    return maeTest, maeTrain, r2Train, r2Test, model, yPredTest

randomForestMaeTestSklearn, randomForestMaeTrainSklearn,randomForestR2TrainSklearn, \
randomForestR2TestSklearn, randomForestModelSklearn, randomForestyPredTestSklearn = createRandomForestModelSklearn()


# 加载模型预测

def loadModelAndPred(data, yTrue=None):
    '''加载预测模型进行验证'''
    lryPredSklearn = joblib.load('./models/lrModelSklearn.pkl').predict(data)
    svryPredSklearn = joblib.load('./models/svrModelSklearn.pkl').predict(data)
    randomForestyPredSklearn = joblib.load('./models/randomForestModelSklearn.pkl').predict(data)

    # 收集预测结果形成csv文件
    allModelPreds = {
                    '线性回归':lryPredSklearn,
                    '支持向量机': svryPredSklearn,
                    '随机森林':randomForestyPredSklearn
    }
    allModelPreds = pd.DataFrame(allModelPreds).round(3)
    if yTrue is not None:
        allModelPreds['真实值'] = yTrue
    allModelPreds.to_csv('./data/allModelPreds2.csv', encoding='gbk')
    return allModelPreds

# 读取带预测数据集
allModelPreds = loadModelAndPred(xTest)
allModelPreds.head()



















