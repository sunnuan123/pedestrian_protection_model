'''
@File  :preprocess_data.py
@Author:SunNuan
@Date  :2023/12/14 11:39
@Desc  :
'''
import pandas  as pd
from sklearn.impute import SimpleImputer
import copy
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold,GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold


def getRawData():
    data1 = pd.read_csv('./data/data_1110.csv', na_values={'dist_out_in':0,'dist_out_mid':0,'dist_mid_in':0,'dist_in_hard':0})
    data2 = pd.read_csv('./data/data_1205.csv',na_values={'dist_out_in':0,'dist_out_mid':0,'dist_mid_in':0,'dist_in_hard':0})
    raw_data = pd.concat([data1,data2], axis=0).iloc[:,1:].reset_index(drop=True)
    return raw_data.iloc[:,:-1], raw_data['HIC15']




def imputerVar(data, label, threshold=1):
    '''缺失处理和方差筛选'''
    data = pd.DataFrame(SimpleImputer().fit_transform(data),columns=data.columns)
    varModel = VarianceThreshold(threshold=threshold).fit(data)
#     print(data.shape)
#     print(set(varModel.feature_names_in_)-set(varModel.get_feature_names_out()))
    data = varModel.transform(data)
    data = pd.DataFrame(data, columns=varModel.get_feature_names_out())
    return data, label


def solveoutliers(flag, data, label, contamination):
    '''是否处理异常值'''
    def isoForest(data, label, contamination=0.1):
        '''孤立森林去除异常值'''
        IsoData = copy.deepcopy(data)
        clf = IsolationForest(max_samples='auto', random_state=0,max_features=1,contamination=contamination).fit(IsoData)
        myIsoDataIndex = clf.predict(IsoData)
        # print(Counter(myIsoDataIndex))
        data_ = IsoData[myIsoDataIndex==1]
        label_ = label[myIsoDataIndex==1]
        iso_data = IsoData[myIsoDataIndex==-1]
        iso_label = label[myIsoDataIndex==-1]
        pd.concat([iso_data,iso_label], axis=1).to_csv('./data/significantSample.csv')
        return data_, label_, iso_data, iso_label
    if flag:
        return isoForest(data, label, contamination)
    return data, label, '', ''


def stdSplitData(data, label):
    '''数据标准化，然后拆分数据集'''
    data = pd.DataFrame(preprocessing.StandardScaler().fit_transform(data), columns=data.columns)
    x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.1,random_state=420)
    return x_train,x_test,y_train,y_test


def getData():
    raw_data, raw_label = getRawData()
    data, label = imputerVar(raw_data, raw_label, threshold=0)
    data, label, iso_data, iso_label = solveoutliers(False, data, label, 0.05)
    data['label'] = label
    data.to_csv('./data/stdData.csv',index=False)
    x_train, x_test, y_train, y_test = stdSplitData(data, label)
    return x_train, x_test, y_train, y_test


if __name__=='__main__':
    getData()

