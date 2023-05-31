import pywt

import pandas as pd
import numpy as np
import time

import math

from sklearn.decomposition import PCA

from nltk import FreqDist

from em_gmm import estimate #导入EM算法估计权重

from tfidfval import count  #导入TF算法估计权重

#---------------------------------------------数据预处理------------------------------------

def process_csv_data(data_folder,lang_type,count_npk):
    # #修补残缺数据
    nlpl_data=pd.read_csv(data_folder+'%s_np_ex_%s_origin.csv'%(lang_type,count_npk),index_col=0,sep='|',encoding='utf-8')
    #去除噪声数据
    nlpl_data[nlpl_data==-10000]=np.nan #将缺省值置换nan  数据爬虫不存在
    nlpl_data[nlpl_data==0]=np.nan      #将缺省值置换nan  帖子本身没给值

    #每列去除噪声点,不具备分布的特征需要去除噪点
    remove_key=['qview','qscore','cscore']  #类别点
    for key in remove_key:
        key_freq=FreqDist([i for i in nlpl_data[key].tolist() if not math.isnan(i)])
        freq_num=[i[0] for i in key_freq.most_common()]        #按频率出现从大到小排序
        key_noise=freq_num[-int(0.01*len(freq_num)):]          #出现最小%1比例作为噪声点,分析数据
        key_data=[i if i not in key_noise else np.nan for i in nlpl_data[key].tolist()]
        key_Serie=pd.Series(key_data, index=nlpl_data.index)
        nlpl_data[key]=key_Serie

    # 每列NaN填充平均数
    column_key=['qview','qanswer','qfavorite','qscore','qcomment','cscore','ccomment']
    #  列属性依次循环
    for key in column_key:
        # 填充数据
        nlpl_data[key]=nlpl_data[key].fillna(round(nlpl_data[key].mean()))

    # 归一化列表
    column_key=['qview','qanswer','qfavorite','qscore','qcomment','cscore','ccomment']
    # 归一化操作
    data_mn=nlpl_data.reindex(columns=column_key).apply(lambda x:(x-x.min())/(x.max()-x.min()),0)
    # 归一化值替换
    nlpl_data[column_key]=data_mn
    # 保存数据
    nlpl_data.to_csv(data_folder+'%s_np_ex_%s_normalized.csv'%(lang_type,count_npk),encoding='utf-8',sep='|')

#---------------------------------------------问题属性的打分------------------------------------
#问题属性小波变换
def query_signl_dwt(signal,wt_type): #4层小波变换 L=4

    LF1, HF1= pywt.dwt(signal,wt_type)   #LF01:低频分量         HF01:高频分量
    LF2, HF2= pywt.dwt(LF1, wt_type)     #LF02:低频分量         HF02:高频分量
    LF3, HF3= pywt.dwt(LF2, wt_type)     #LF03:低频分量         HF03:高频分量
    LF4, HF4= pywt.dwt(LF3, wt_type)     #LF04:低频分量         HF04:高频分量

    return HF1,HF2,HF3,HF4,LF4

#问题属性小波的系数向量集
def query_em_fuse(C1,C2,C3,C4,C5):
    #估计线性融合时的权重
    w1,w2,w3,w4,w5=estimate(C1.tolist()),estimate(C2.tolist()),estimate(C3.tolist()),estimate(C4.tolist()),estimate(C5.tolist())

    #K维特征融合后的系数集
    new_HF1=[w1[i]*C1[:,i] for i in range(C1.shape[1])][0]
    new_HF2=[w2[i]*C2[:,i] for i in range(C2.shape[1])][0]
    new_HF3=[w3[i]*C3[:,i] for i in range(C3.shape[1])][0]
    new_HF4=[w4[i]*C4[:,i] for i in range(C4.shape[1])][0]
    new_LF4=[w5[i]*C5[:,i] for i in range(C5.shape[1])][0]

    return new_HF1,new_HF2,new_HF3,new_HF4,new_LF4 #作为新的低频分量

#问题属性小波逆变换
def query_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type): #4层小波逆变换 L=4

    new_LF3=pywt.idwt(new_LF4,new_HF4,wt_type)   # (低频分量,高频分量)---→上一级(低频分量)
    new_LF2=pywt.idwt(new_LF3,new_HF3,wt_type)
    new_LF1=pywt.idwt(new_LF2,new_HF2,wt_type)
    new_signal= pywt.idwt(new_LF1,new_HF1,wt_type)

    return new_signal

#问题融合打分
def query_dwt_score(query_feat,wt_type):

    signl0,signl1,signl2,signl3,signl4=query_feat[:,0],query_feat[:,1],query_feat[:,2],query_feat[:,3],query_feat[:,4]
    #小波变换
    HF01,HF02,HF03,HF04,LF04=query_signl_dwt(signl0,wt_type) #第一棵树
    HF11,HF12,HF13,HF14,LF14=query_signl_dwt(signl1,wt_type) #第二棵树
    HF21,HF22,HF23,HF24,LF24=query_signl_dwt(signl2,wt_type) #第三棵树
    HF31,HF32,HF33,HF34,LF34=query_signl_dwt(signl3,wt_type) #第四棵树
    HF41,HF42,HF43,HF44,LF44=query_signl_dwt(signl4,wt_type) #第五棵树
    #构建系数矩阵
    C1=np.concatenate((HF01.reshape(-1,1),HF11.reshape(-1,1),HF21.reshape(-1,1),HF31.reshape(-1,1),HF41.reshape(-1,1)),axis=1)
    C2=np.concatenate((HF02.reshape(-1,1),HF12.reshape(-1,1),HF22.reshape(-1,1),HF32.reshape(-1,1),HF42.reshape(-1,1)),axis=1)
    C3=np.concatenate((HF03.reshape(-1,1),HF13.reshape(-1,1),HF23.reshape(-1,1),HF33.reshape(-1,1),HF43.reshape(-1,1)),axis=1)
    C4=np.concatenate((HF04.reshape(-1,1),HF14.reshape(-1,1),HF24.reshape(-1,1),HF34.reshape(-1,1),HF44.reshape(-1,1)),axis=1)
    C5=np.concatenate((LF04.reshape(-1,1),LF14.reshape(-1,1),LF24.reshape(-1,1),LF34.reshape(-1,1),LF44.reshape(-1,1)),axis=1)
    #每个系数矩阵的维度不同

    new_HF1,new_HF2,new_HF3,new_HF4,new_LF4=query_em_fuse(C1,C2,C3,C4,C5)
    new_signal=query_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type)

    return new_signal #问题属性的综合打分

#---------------------------------------------答案属性的打分------------------------------------
#答案属性小波变换
def code_signl_dwt(signal,wt_type): #4层小波变换 L=4
    LF1, HF1= pywt.dwt(signal,wt_type)  #LF01:低频分量         HF01:高频分量
    LF2, HF2= pywt.dwt(LF1,wt_type)     #LF02:低频分量         HF02:高频分量
    LF3, HF3= pywt.dwt(LF2,wt_type)     #LF03:低频分量         HF03:高频分量
    LF4, HF4= pywt.dwt(LF3,wt_type)     #LF04:低频分量         HF04:高频分量
    return HF1,HF2,HF3,HF4,LF4

#答案属性小波逆变换
def code_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type): #4层小波逆变换 L=4
    new_LF3=pywt.idwt(new_LF4,new_HF4,wt_type)   # (低频分量,高频分量)---→上一级(低频分量)
    new_LF2=pywt.idwt(new_LF3,new_HF3,wt_type)
    new_LF1=pywt.idwt(new_LF2,new_HF2,wt_type)
    new_signal=pywt.idwt(new_LF1,new_HF1,wt_type)

    return new_signal

#答案属性小波的系数向量集
def code_em_fuse(C1,C2,C3,C4,C5):
    #估计线性融合时的权重
    w1,w2,w3,w4,w5=estimate(C1.tolist()),estimate(C2.tolist()),estimate(C3.tolist()),estimate(C4.tolist()),estimate(C5.tolist())
    #K维特征融合后的系数集
    new_HF1=[w1[i]*C1[:,i] for i in range(C1.shape[1])][0]
    new_HF2=[w2[i]*C2[:,i] for i in range(C2.shape[1])][0]
    new_HF3=[w3[i]*C3[:,i] for i in range(C3.shape[1])][0]
    new_HF4=[w4[i]*C4[:,i] for i in range(C4.shape[1])][0]
    new_LF4=[w5[i]*C5[:,i] for i in range(C5.shape[1])][0]

    return new_HF1,new_HF2,new_HF3,new_HF4,new_LF4 #作为新的低频分量

#答案融合打分
def code_dwt_score(code_feat,wt_type):
    signl0,signl1=code_feat[:,0],code_feat[:,1]
    #小波变换
    HF01,HF02,HF03,HF04,LF04=code_signl_dwt(signl0,wt_type) #第一棵树
    HF11,HF12,HF13,HF14,LF14=code_signl_dwt(signl1,wt_type) #第二棵树
    #构建系数矩阵
    C1=np.concatenate((HF01.reshape(-1,1),HF11.reshape(-1,1)),axis=1)
    C2=np.concatenate((HF02.reshape(-1,1),HF12.reshape(-1,1)),axis=1)
    C3=np.concatenate((HF03.reshape(-1,1),HF13.reshape(-1,1)),axis=1)
    C4=np.concatenate((HF04.reshape(-1,1),HF14.reshape(-1,1)),axis=1)
    C5=np.concatenate((LF04.reshape(-1,1),LF14.reshape(-1,1)),axis=1)

    #每个系数矩阵的维度不同
    new_HF1,new_HF2,new_HF3,new_HF4,new_LF4=code_em_fuse(C1,C2,C3,C4,C5)
    new_signal=code_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type)

    return new_signal #答案属性的综合打分


#---------------------------------------------全体属性的打分------------------------------------
#全体属性小波变换
def allqc_signl_dwt(signal,wt_type): #4层小波变换 L=4
    LF1, HF1= pywt.dwt(signal,wt_type)   #LF01:低频分量         HF01:高频分量
    LF2, HF2= pywt.dwt(LF1, wt_type)     #LF02:低频分量         HF02:高频分量
    LF3, HF3= pywt.dwt(LF2, wt_type)     #LF03:低频分量         HF03:高频分量
    LF4, HF4= pywt.dwt(LF3, wt_type)     #LF04:低频分量         HF04:高频分量
    return HF1,HF2,HF3,HF4,LF4

#全体属性小波的系数向量集
def allqc_em_fuse(C1,C2,C3,C4,C5):
    #估计线性融合时的权重
    w1,w2,w3,w4,w5=estimate(C1.tolist()),estimate(C2.tolist()),estimate(C3.tolist()),estimate(C4.tolist()),estimate(C5.tolist())
    #K维特征融合后的系数集
    new_HF1=[w1[i]*C1[:,i] for i in range(C1.shape[1])][0]
    new_HF2=[w2[i]*C2[:,i] for i in range(C2.shape[1])][0]
    new_HF3=[w3[i]*C3[:,i] for i in range(C3.shape[1])][0]
    new_HF4=[w4[i]*C4[:,i] for i in range(C4.shape[1])][0]
    new_LF4=[w5[i]*C5[:,i] for i in range(C5.shape[1])][0]
    return new_HF1,new_HF2,new_HF3,new_HF4,new_LF4 #作为新的低频分量

#全体属性小波逆变换
def allqc_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type): #4层小波逆变换 L=4
    new_LF3=pywt.idwt(new_LF4,new_HF4,wt_type)   # (低频分量,高频分量)---→上一级(低频分量)
    new_LF2=pywt.idwt(new_LF3,new_HF3,wt_type)
    new_LF1=pywt.idwt(new_LF2,new_HF2,wt_type)
    new_signal= pywt.idwt(new_LF1,new_HF1,wt_type)
    return new_signal

#全体融合打分
def allqc_dwt_score(all_feat,wt_type):
    signl0,signl1,signl2,signl3,signl4,signl5,signl6=all_feat[:,0],all_feat[:,1],all_feat[:,2],all_feat[:,3],all_feat[:,4],all_feat[:,5],all_feat[:,6]
    #小波变换
    HF01,HF02,HF03,HF04,LF04=allqc_signl_dwt(signl0,wt_type) #第一棵树
    HF11,HF12,HF13,HF14,LF14=allqc_signl_dwt(signl1,wt_type) #第二棵树
    HF21,HF22,HF23,HF24,LF24=allqc_signl_dwt(signl2,wt_type) #第三棵树
    HF31,HF32,HF33,HF34,LF34=allqc_signl_dwt(signl3,wt_type) #第四棵树
    HF41,HF42,HF43,HF44,LF44=allqc_signl_dwt(signl4,wt_type) #第五棵树
    HF51,HF52,HF53,HF54,LF54=allqc_signl_dwt(signl5,wt_type) #第六棵树
    HF61,HF62,HF63,HF64,LF64=allqc_signl_dwt(signl6,wt_type) #第七棵树
    #构建系数矩阵
    C1=np.concatenate((HF01.reshape(-1,1),HF11.reshape(-1,1),HF21.reshape(-1,1),HF31.reshape(-1,1),HF41.reshape(-1,1),HF51.reshape(-1,1),HF61.reshape(-1,1)),axis=1)
    C2=np.concatenate((HF02.reshape(-1,1),HF12.reshape(-1,1),HF22.reshape(-1,1),HF32.reshape(-1,1),HF42.reshape(-1,1),HF52.reshape(-1,1),HF62.reshape(-1,1)),axis=1)
    C3=np.concatenate((HF03.reshape(-1,1),HF13.reshape(-1,1),HF23.reshape(-1,1),HF33.reshape(-1,1),HF43.reshape(-1,1),HF53.reshape(-1,1),HF63.reshape(-1,1)),axis=1)
    C4=np.concatenate((HF04.reshape(-1,1),HF14.reshape(-1,1),HF24.reshape(-1,1),HF34.reshape(-1,1),HF44.reshape(-1,1),HF54.reshape(-1,1),HF64.reshape(-1,1)),axis=1)
    C5=np.concatenate((LF04.reshape(-1,1),LF14.reshape(-1,1),LF24.reshape(-1,1),LF34.reshape(-1,1),LF44.reshape(-1,1),LF54.reshape(-1,1),LF64.reshape(-1,1)),axis=1)
    #每个系数矩阵的维度不同
    new_HF1,new_HF2,new_HF3,new_HF4,new_LF4=allqc_em_fuse(C1,C2,C3,C4,C5)
    new_signal=allqc_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type)

    return new_signal #全体属性的综合打分


#全体融合打分
def pcaqc_dwt_score(pca_feat,wt_type):
    signl0,signl1,signl2,signl3,signl4=pca_feat[:,0],pca_feat[:,1],pca_feat[:,2],pca_feat[:,3],pca_feat[:,4]
    #小波变换
    HF01,HF02,HF03,HF04,LF04=allqc_signl_dwt(signl0,wt_type) #第一棵树
    HF11,HF12,HF13,HF14,LF14=allqc_signl_dwt(signl1,wt_type) #第二棵树
    HF21,HF22,HF23,HF24,LF24=allqc_signl_dwt(signl2,wt_type) #第三棵树
    HF31,HF32,HF33,HF34,LF34=allqc_signl_dwt(signl3,wt_type) #第四棵树
    HF41,HF42,HF43,HF44,LF44=allqc_signl_dwt(signl4,wt_type) #第五棵树
    #构建系数矩阵
    C1=np.concatenate((HF01.reshape(-1,1),HF11.reshape(-1,1),HF21.reshape(-1,1),HF31.reshape(-1,1),HF41.reshape(-1,1)),axis=1)
    C2=np.concatenate((HF02.reshape(-1,1),HF12.reshape(-1,1),HF22.reshape(-1,1),HF32.reshape(-1,1),HF42.reshape(-1,1)),axis=1)
    C3=np.concatenate((HF03.reshape(-1,1),HF13.reshape(-1,1),HF23.reshape(-1,1),HF33.reshape(-1,1),HF43.reshape(-1,1)),axis=1)
    C4=np.concatenate((HF04.reshape(-1,1),HF14.reshape(-1,1),HF24.reshape(-1,1),HF34.reshape(-1,1),HF44.reshape(-1,1)),axis=1)
    C5=np.concatenate((LF04.reshape(-1,1),LF14.reshape(-1,1),LF24.reshape(-1,1),LF34.reshape(-1,1),LF44.reshape(-1,1)),axis=1)

    #每个系数矩阵的维度不同
    new_HF1,new_HF2,new_HF3,new_HF4,new_LF4=allqc_em_fuse(C1,C2,C3,C4,C5)
    new_signal=allqc_signl_idwt(new_HF1,new_HF2,new_HF3,new_HF4,new_LF4,wt_type)

    return new_signal #全体属性的综合打分


#---------------------------------------------   打分融合------------------------------------
def mean_fuse_score(all_data):
    #估计线性融合时的权重
    w=[0.5,0.5]
    #K维特征融合后的系数集
    mean_score=sum([w[i]*all_data[:,i] for i in range(2)])
    return mean_score

def allqc_matr_score(all_data):
    #估计线性融合时的权重
    w=estimate(all_data.tolist())
    #K维特征融合后的系数集
    allme_score=sum([w[i]*all_data[:,i] for i in range(7)])
    return allme_score


def code_query_score(data_folder,lang_type,count_k): #取前M个作为高质量的语料

    # K=4 4维
    score_data=pd.read_csv(data_folder+'%s_np_ex_%s_normalized.csv'%(lang_type,count_k),index_col=0,sep='|',encoding='utf-8')

    #保证4层分解
    data_frame=score_data.iloc[:count_k]

    # 5个特征
    query_feat=data_frame.reindex(columns=['qview','qanswer','qfavorite','qscore','qcomment']).values

    # 2个特征
    code_feat=data_frame.reindex(columns=['cscore','ccomment']).values

    # 7个特征
    all_feat=data_frame.reindex(columns=['qview','qanswer','qfavorite','qscore','qcomment','qscore','qcomment']).values


    #-----------------------------------打分融合--------------------------------
    #直接融合
    start1_time = time.time()
    df_score=allqc_matr_score(all_feat)
    print(len(df_score))                                          #c#4000,sql4000;  c#20000, sql20000
    end1_time = time.time()
    print("df_score %.2fs"%round(end1_time-start1_time,2))       #115.10s，117.14s，15.16s，14.09s

    #单重小波
    start2_time = time.time()
    hw_score=allqc_dwt_score(all_feat,'haar')
    print(len(hw_score))
    end2_time = time.time()
    print("hw_score %.2fs"%round(end2_time-start2_time,2))       #117.07s，116.36s，338.30s，179.93s

    #单重小波
    start3_time = time.time()
    dw_score=allqc_dwt_score(all_feat,'db1')
    print(len(dw_score))
    end3_time = time.time()
    print("dw_score %.2fs" % round(end3_time - start3_time, 2))   #117.49s，117.14s，244.58s，223.81s

    #双重小波
    start4_time = time.time()
    qu_score=query_dwt_score(query_feat,'haar')
    co_score=code_dwt_score(code_feat,'haar')
    all_data=np.concatenate((qu_score.reshape(-1,1),co_score.reshape(-1,1)),axis=1)
    ab_score=mean_fuse_score(all_data)
    print(len(ab_score))
    end4_time = time.time()
    print("ab_score %.2fs"%round(end4_time-start4_time,2))       #85.01s，85.91s，134.01s，119.93s

    #tfidf权重
    start5_time = time.time()
    query_data = data_frame.reindex(columns=['orgin_query']).values
    query_list = [line[0] for line in query_data.tolist()]
    tf_score = count(query_list)
    print(len(tf_score))
    end5_time = time.time()
    print("tf_score %.2fs"%round(end5_time-start5_time,2))       #20.73s，20.23s，403.12s，395.23s

    #PCA变换
    start6_time = time.time() #原特征7个
    pca = PCA(n_components=5) #变特征5个
    pca_feat=pca.fit_transform(all_feat)
    pc_score=pcaqc_dwt_score(pca_feat,'haar')
    print(len(pc_score))
    end6_time = time.time()
    print("pc_score %.2fs"%round(end6_time-start6_time,2))       #83.77s，78.13s，83.77s，93.34s

    #-----------------------------------打分融合--------------------------------
    size=data_frame.shape[1] #统计列数

    #直接特征变换 SN-DFF
    data_frame.insert(size+0, 'SN-DFF', df_score)
    #综合全体属性，SN-WTFF-HW
    data_frame.insert(size+1, 'SN-WTFF-HW', hw_score)
    #综合全体属性，SN-WTFF-DW
    data_frame.insert(size+2, 'SN-WTFF-DW', dw_score)
    #综合问题-答案，DB-WTFF-AB
    data_frame.insert(size+3, 'DB-WTFF-AB', ab_score)
    #综合查询属性，TF-IDF
    data_frame.insert(size+4, 'TF-IDF', tf_score)
    #综合全体属性，PC-SN-WTFF
    data_frame.insert(size+5, 'PC-SN-WTFF', pc_score)

    #保存
    data_frame.to_csv(data_folder+'%s_np_ex_%s_scored.csv'%(lang_type,count_k),sep='|',encoding='utf-8')

#---------------------------------------------主程序------------------------------------

# 参数配置
data_folder ='./corpus_test/'

csharp_type ='c#'
sqlang_type ='sql'

count1_npk = 4000
count2_npk = 20000


if __name__ == '__main__':
    # c# 4000
    process_csv_data(data_folder,csharp_type,count1_npk)
    code_query_score(data_folder,csharp_type ,count1_npk)
    # c# 20000
    process_csv_data(data_folder,csharp_type ,count2_npk)
    code_query_score(data_folder,csharp_type ,count2_npk)
    # sql 4000
    process_csv_data(data_folder,sqlang_type,count1_npk)
    code_query_score(data_folder,sqlang_type,count1_npk)
    # sql 20000
    process_csv_data(data_folder,sqlang_type,count2_npk)
    code_query_score(data_folder,sqlang_type,count2_npk)
