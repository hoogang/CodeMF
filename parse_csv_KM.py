import pywt
import math
import pandas as pd
import numpy as np
from nltk import FreqDist
#导入EM算法估计权重
from em_gmm import estimate

#---------------------------------------------全体属性的打分------------------------------------
##############################################选用SN-WTFF-HW框架打分##########################3
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


def wtff_for_data(attr_data,data_folder,csvda_type,lang_type):
    # 修补残缺数据，去除噪声数据
    #缺省值补全
    attr_data[attr_data==-10000]=np.nan #将缺省值置换nan  数据爬虫不存在
    attr_data[attr_data==0]=np.nan      #将缺省值置换nan  帖子本身没给值

    #每列去除噪声点,不具备分布的特征需要去除噪点
    remove_key=['qview','qscore','cscore']  #类别点
    for key in remove_key:
        key_freq=FreqDist([i for i in attr_data[key].tolist() if not math.isnan(i)])
        freq_num=[i[0] for i in key_freq.most_common()]  #按频率出现从大到小排序
        key_noise=freq_num[-int(0.01*len(freq_num)):]    #出现最小%1比例作为噪声点,分析数据
        key_data=[i if i not in key_noise else np.nan for i in attr_data[key].tolist()]
        key_Serie=pd.Series(key_data, index=attr_data.index)
        attr_data[key]=key_Serie

    #每列NaN填充平均数
    column_key=['qview','qanswer','qfavorite','qscore','qcomment','cscore','ccomment']
    #列属性依次循环
    for key in column_key:
        #  填充数据
        attr_data[key]=attr_data[key].fillna(round(attr_data[key].mean()))

    #归一化列表
    column_key=['qview','qanswer','qfavorite','qscore','qcomment','cscore','ccomment']
    #数据归一化操作
    data_mn=attr_data.reindex(columns=column_key).apply(lambda x:(x-x.min())/(x.max()-x.min()),0)
    #相应的值替换
    attr_data[column_key]=data_mn
    # ---------------------------------------------数据预处理------------------------------------

    # ---------------------------------------------数据需截断------------------------------------
    #数据长度保证是2^x样式才能分解
    attr_length = len(attr_data)
    print('%s类型的%s数据最开始爬取的原始长度是%d'%(csvda_type,lang_type,attr_length))
    #开始循环查找，数据长度
    lang_cut= math.floor(attr_length/(2**4))*(2**4)
    print('%s类型的%s数据用小波变换的截断长度是%d'%(csvda_type,lang_type,lang_cut))
    '''  
    单选      csharp   sqlang   javang    python 
        开始  171773   211215   156432    150190
        截断  171760   211200   156432    150176
    
    多选      csharp   sqlang   javang    python 
        开始  81746    90087    73001     86722 
        截断  81744    90080    72992     86720
    '''
    #保证4层可以分解
    attr_data = attr_data.iloc[:lang_cut]
    #---------------------------------------------数据需截断------------------------------------

    #提取特征属性
    column_key = ['qview', 'qanswer', 'qfavorite', 'qscore', 'qcomment', 'cscore', 'ccomment']
    attr_feat = attr_data.reindex(columns=column_key).values
    #特征融合打分
    fuse_score = allqc_dwt_score(attr_feat,'haar')
    print('##############%s类型的%s数据融合打分结束！############'%(csvda_type,lang_type))
    #融合打分赋值
    attr_data.insert(attr_data.shape[1], 'fuse_score', fuse_score)
    #打分值排序，从大到小排序
    sort_attr = attr_data.sort_values(by='fuse_score', ascending=False)
    #重建0-K索引值
    sort_attr = sort_attr.reset_index(drop=True)
    #抽取前项索引值
    sort_attr = sort_attr.reindex(columns=['acceptID'])
    #保存前项索引值
    sort_attr.to_csv(data_folder+'%s_%s_np_rid.csv'%(csvda_type,lang_type),sep='|',encoding='utf-8')


def lang_rank_data(data_folder,csvda_type,lang_type):

    # ---------------------------------------------数据预处理------------------------------------
    if csvda_type == 'single':
        # 加载数据
        nlpl_data = pd.read_csv(data_folder + 'single_%s_np.csv'%lang_type, index_col=0, sep='|', encoding='utf-8')
        # 抽取社交值
        save_key = ['acceptID', 'qview', 'qanswer', 'qfavorite', 'qscore', 'qcomment', 'cscore', 'ccomment']
        # 保存数据
        attr_data = nlpl_data.reindex(columns=save_key)
        # 数据排名
        wtff_for_data(attr_data, data_folder, csvda_type, lang_type)

    # ---------------------------------------------数据预处理------------------------------------
    if csvda_type == 'mutiple':
        # 加载数据
        nlpl_data = pd.read_csv(data_folder + 'mutiple_%s_np.csv'%lang_type, index_col=0, sep='|', encoding='utf-8')
        nlpl_data = nlpl_data[nlpl_data['code_rank']==0]
        # 抽取社交值
        save_key = ['acceptID', 'qview', 'qanswer', 'qfavorite', 'qscore', 'qcomment', 'cscore', 'ccomment']
        # 保存数据
        attr_data = nlpl_data.reindex(columns=save_key)
        # 数据排名
        wtff_for_data(attr_data, data_folder, csvda_type, lang_type)


#---------------------------------------------主程序------------------------------------
def get_topk_data(data_folder,lang_type,save_folder):

    ######################################单选数据处理###########################################
    # 接受ID索引值
    single_pid = pd.read_csv(data_folder+'single_%s_np_rid.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    # 给出索引
    single_idx = single_pid['acceptID'].tolist()

    # 找出前项topk值
    single_data = pd.read_csv(data_folder+'single_%s_np.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    # 抽取指定的acceid值
    single_data = single_data[single_data['acceptID'].isin(single_idx)]
    print('%s筛选之前的大小是%d'%(lang_type,len(single_data)))
    # 按指定的列排序
    single_data['acceptID'] = single_data['acceptID'].astype('category')
    single_data['acceptID'].cat.set_categories(single_idx, inplace=True)
    single_data.sort_values('acceptID', ascending=True)
    # 重建0-K索引值
    single_data = single_data.reset_index(drop=True)
    # 打分完抽取
    save_key = ['ID','orgin_query','qcont','orgin_code','ccont','code_rank']
    # 抽取非属性列
    single_data = single_data.reindex(columns=save_key)
    print('%s单选数据的大小是%d'%(lang_type,len(single_data))) #csharp:171760,sqlang:211200,javang:156432,python:150176
    # 保存前项数据
    single_data.to_csv(save_folder+'single_fuse_%s_np.csv'%lang_type,sep='|',encoding='utf-8')
    print('%s单选数据融合排序结束！'%lang_type)

    '''  
    单选      csharp   sqlang   javang    python 
             171760   211200    156432   150176 
    多选      csharp   sqlang   javang    python 
             240888   259902    214896   285330
    '''

    ######################################多选数据处理###########################################
    # 接受ID索引值
    mutiple_pid = pd.read_csv(data_folder+'mutiple_%s_np_rid.csv'%lang_type,index_col=0, sep='|',encoding='utf-8')
    # 给出索引
    mutiple_idx = mutiple_pid['acceptID'].tolist()

    # 找出前项topk值
    mutiple_data = pd.read_csv(data_folder+'mutiple_%s_np.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    # 抽取指定的acceid值
    mutiple_data = mutiple_data[mutiple_data['acceptID'].isin(mutiple_idx)]
    # 按指定的列排序 pandas 1.2
    mutiple_data['acceptID'] = mutiple_data['acceptID'].astype('category')
    mutiple_data['acceptID'].cat.set_categories(mutiple_idx, inplace=True)
    mutiple_data.sort_values('acceptID', ascending=True)

    # 分组排序(频次顺序变了）rank  0-k
    mutiple_data = mutiple_data.groupby('acceptID').apply(lambda row: row.sort_values(by='code_rank'))

    # 重建0-K索引值
    mutiple_data = mutiple_data.reset_index(drop=True)
    # 打分完抽取
    save_key = ['ID', 'orgin_query', 'qcont', 'orgin_code', 'ccont','code_rank']
    # 抽取非属性列
    mutiple_data = mutiple_data.reindex(columns=save_key)
    print('%s多选数据的大小是%d'%(lang_type,len(mutiple_data))) #csharp:240888,sqlang:259902,javang:214896,python:285330
    # 保存前项数据
    mutiple_data.to_csv(save_folder+'mutiple_fuse_%s_np.csv'%lang_type,sep='|',encoding='utf-8')
    print('%s多选数据融合排序结束！'%lang_type)


# 参数配置
data_folder ='./corpus_csv/'
save_folder ='../rank_corpus/'

single_type = 'single'
mutiple_type= 'mutiple'

csharp_type = 'csharp'

sqlang_type = 'sqlang'

javang_type = 'javang'

python_type = 'python'


if __name__ == '__main__':
    #csharp
    #lang_rank_data(data_folder,single_type,csharp_type)
    #lang_rank_data(data_folder,mutiple_type,csharp_type)
    get_topk_data(data_folder,csharp_type,save_folder)

    # sqlang
    #lang_rank_data(data_folder,single_type,sqlang_type)
    #lang_rank_data(data_folder,mutiple_type,sqlang_type)
    get_topk_data(data_folder,sqlang_type,save_folder)

    # javang
    #lang_rank_data(data_folder,single_type,javang_type)
    #lang_rank_data(data_folder,mutiple_type,javang_type)
    get_topk_data(data_folder,javang_type,save_folder)

    # python
    #lang_rank_data(data_folder,single_type,python_type)
    #lang_rank_data(data_folder,mutiple_type,python_type)
    get_topk_data(data_folder,python_type,save_folder)
