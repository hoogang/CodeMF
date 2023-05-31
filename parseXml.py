#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

import pandas as pd
from copy import copy

#分类标签
from svm import Classfier
#解析数据
from bs4 import BeautifulSoup


#--------------------------linux代码，配置可执行-----------------------
def extract_xml(data_folder,csharp_name,sqlang_name,javang_name,python_name,answer_xml):

    #grep  正则表达式  #提取xml中c#标签
    cmd_csharp = 'egrep  "c#" %s > %s'%(data_folder+'posts_2018.xml',data_folder+csharp_name)
    print(cmd_csharp)
    os.popen(cmd_csharp).read()

    #grep  正则表达式  #提取xml中sql标签
    cmd_sqlang = 'egrep "sql"  %s  > %s'%(data_folder+'posts_2018.xml',data_folder+sqlang_name)
    print(cmd_sqlang)
    os.popen(cmd_sqlang).read()

    #grep  正则表达式  #提取xml中java标签
    cmd_javang = 'egrep \'.*Tags=".*&lt;java&gt.*"\' %s > %s'%(data_folder+'posts_2018.xml',data_folder+javang_name)
    print(cmd_javang)
    os.popen(cmd_javang).read()

    #grep 正则表达式  #提取xml中python标签
    cmd_python = 'egrep \'.*Tags=".*&lt;python&gt.*"\' %s > %s'%(data_folder+'posts_2018.xml',data_folder+python_name)
    print(cmd_python)
    os.popen(cmd_python).read()

    #grep 正则表达式  #所有回答xml数据
    cmd_answer = 'egrep \'PostTypeId="2"\' %s > %s'%(data_folder+'answer_2018.xml',data_folder +answer_xml)
    os.popen(cmd_answer).read()



#分类器
s = Classfier()
s.train("./balanced/pos_train.txt", "./balanced/neg_train.txt")
s.test("./balanced/pos_test.txt", "./balanced/neg_test.txt")


# --------------------------解析XML文件的标签-----------------------------#
#滤除非法字符(不滤除后续解析出错)
def filter_inva_char(line):
    #去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|%|_|,|\'|:|=|>|<|\"|;|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


#解析属性标签
def get_tag_param(rowTag,tag):
    try:
        output = rowTag[tag]
        return output
    except KeyError:
        return '-10000'

#获取code中数据（满足条件）,取多个候选
def parse_all_code(codeTag):
    #  候选代码
    mucodes = []

    if len(codeTag) >= 1:
         #  代码一个或多个标签
        for oneTag in codeTag:
            code = oneTag.get_text().strip()
            #  代码片段10-1000字符,过滤字符长度
            if (len(code) >= 10 and len(code) <= 1000):
                #  过滤代码的关键片段
                if code[0] == '<' or code[0] == '=' or code[0] == '@'  or code[0] == '$' or \
                    code[0:7].lower() == 'select' or code[0:7].lower() == 'update' or code[0:6].lower() == 'alter' or \
                    code[0:2].lower() == 'c:' or code[0:4].lower() == 'http' or code[0:4].lower() == 'hkey' or \
                    re.match(r'^[a-zA-Z0-9_]*$', code) is not None:
                    #  加入代码
                    mucodes.append('-1000')
                else:
                    code = re.sub('\n+', '\n', re.sub(' +', ' ', filter_inva_char(code))).strip('\n').replace('\n','\\n')
                    #  满足要求
                    mucodes.append(code)
            else:
                #  代码长度不满足要求
                mucodes.append('-1000')
    else:
        # 不存在
        mucodes.append('-1000')

    return  mucodes


#存储query的数据
def process_xmldata(data_folder,lang_type,answer_xml,split_num,save_folder):

    ################################################找寻有用的query#########################################
    #帖子id，接受id
    qidnum_lis,accept_lis=[],[]
    #打分数
    qscore_lis=[]
    #浏览数、回答数、评论数、喜欢数
    qview_lis,qanswer_lis,qcomment_lis,qfavorite_lis=[],[],[],[]
    #标题
    query_lis=[]
    #contxet
    qcont_lis=[]
    #计数
    count1=0
    count2=0
    with open(data_folder+'%s_tag.xml'%lang_type,'r',encoding='utf-8') as f:
        for line in f:
            count1+=1
            if count1 %10000==0: #总计
                print('查询阶段已经处理--%d条数据'%count1)
            rowTag=BeautifulSoup(line,'lxml').row #row标签
            #标题
            query=get_tag_param(rowTag,'title')
            #  判断查询是否满足要求
            if query!='-10000':  #存在
                #过滤查询query
                queryFilter = s.filter(query)
                if queryFilter==0:
                    accept=get_tag_param(rowTag,'acceptedanswerid')
                    if accept!= '-10000' and lang_type in get_tag_param(rowTag, "tags") :  # 存在
                        count2+=1
                        if count2 % 10000==0:
                            print('查询阶段有效已经处理-------------%d条数据'%count2)
                        # 查询query # 分隔符'|'替换
                        query = filter_inva_char(query)
                        #查询query
                        query_lis.append(query)
                        #接受id
                        accept_lis.append(int(accept))
                        #帖子id
                        qidnum=get_tag_param(rowTag,'id')
                        qidnum_lis.append(int(qidnum))
                        #打分数
                        qscorenum=get_tag_param(rowTag,'score')
                        qscore_lis.append(int(qscorenum))
                        #浏览数
                        qviewnum=get_tag_param(rowTag,'viewcount')
                        qview_lis.append(int(qviewnum))
                        #回答数
                        qanswernum=get_tag_param(rowTag,'answercount')
                        qanswer_lis.append(int(qanswernum))
                        #评论数
                        qcomment=get_tag_param(rowTag,'commentcount')
                        qcomment_lis.append(int(qcomment))
                        #喜欢数
                        qfavorite=get_tag_param(rowTag,'favoritecount')
                        qfavorite_lis.append(int(qfavorite))
                        #body属性
                        body=get_tag_param(rowTag,'body')
                        soup=BeautifulSoup(body,'lxml')
                        #查询context
                        qcont=soup.findAll(text=True)
                        #去除噪音 分隔符'|'替换
                        qcont = ' '.join([filter_inva_char(i) for i in qcont])
                        qcont = re.sub('\n+', '\n', re.sub(' +', ' ', qcont)).strip('\n').replace('\n', '\\n')
                        qcont_lis.append(qcont)

    print('查询阶段有效总计处理-------------%d条数据'%count2)

    print('###################################查询阶段执行完毕！###################################')

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度 #
    index_num = len(accept_lis)  # csharp:360381，sqlang    :,javang:  ,python:
    print('回复中答案id索引的长度-----%d' % index_num)

    # 从小到大的序号排序
    sort_lis = sorted(accept_lis)
    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_lis[i:i + split_num] for i in range(0, len(sort_lis), split_num)]

    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_cors.xml'%lang_type.replace('.xml', 'Part%d.xml' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str,data_folder+answer_xml, data_folder+epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i+1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_cors.xml'%lang_type)

    # 执行命令
    os.popen(ping_cmd).read()     #得到change_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_cors.xml'%lang_type.replace('.xml', 'Part*.xml')
    os.popen(remove_cmd).read()
    print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的change_xml文件#######################################
    # 帖子id
    cidnum_lis = ['-100']*index_num    #-100 没找到
    # 打分数
    cscore_lis = ['-100']*index_num    #-100没找到
    # 评论数
    ccomment_lis = ['-100']*index_num  #-100 没找到
    # 代码
    code_lis = ['-100']*index_num      #-100 没找到
    # context
    ccont_lis = ['-100']*index_num     #-100 没找到
    # 候选排名
    crank_lis = ['-100']*index_num     #-100 没找到

    mucodes_dict = {}
    #计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_cors.xml'%lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:
                print('回复阶段已经处理--%d条数据' % count1)
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            qid = int(line[(sta + 1):end])
            #  还是要判断
            if  qid  in accept_lis:
                # 并不一定从小到大排序且可能accept并没有完全遍历
                id= accept_lis.index(qid)
                #row标签
                rowTag = BeautifulSoup(line,'lxml').row
                #帖子id
                cidnum = get_tag_param(rowTag,'id')
                cidnum_lis[id] = int(cidnum)
                #打分数
                cscorenum = get_tag_param(rowTag,'score')
                cscore_lis[id] = int(cscorenum)
                #评论数
                ccomment = get_tag_param(rowTag,'commentcount')
                ccomment_lis[id] = int(ccomment)
                #body属性
                body = get_tag_param(rowTag,'body')
                soup = BeautifulSoup(body,'lxml')
                #代码code
                codeTag = soup.find_all('code')    # code标签
                # 至少有一个或多个标签
                mucodes = parse_all_code(codeTag)  # code文本
                # 取出第一个代码
                code_lis[id] = mucodes[0]
                # 候选排名
                crank_lis[id] = 0  # 第一个是接受的答案

                # 判断候选个数，最佳代码存在
                if len(mucodes) > 1 and mucodes[0] != '-1000':
                    # 去除非条件候选,更新候选集
                    mucodes = [code for code in mucodes if code and code != '-1000']
                    # 至少第二个候选满足条件，其他候选不作参考
                    if len(mucodes) > 1:
                        count2 += 1
                        if count2 % 10000 == 0:
                            print('候选的有效id已经存储-------------%d条数据' % count2)
                        # 存入候选代码,不是按从小到大,索引位置：候选列表，候选长度
                        mucodes_dict[id] = (mucodes[1:], len(mucodes)-1)

                #代码context
                ccont = soup.findAll(text=True)
                #去除噪音  分隔符'|'替换
                ccont = ' '.join([filter_inva_char(i) for i in ccont])
                ccont = re.sub('\n+', '\n', re.sub(' +', ' ', ccont)).strip('\n').replace('\n','\\n')
                ccont_lis[id]=ccont


    print('回复阶段有效总计处理-------------%d条数据'%count1)
    print('回复阶段有效候选总计-------------%d条数据'%count2)

    print('###################################回复阶段执行完毕！###################################')

    ##############################查询的数据###############################
    # 查询帖子id  ！！必须采用拷贝机制
    query_idnum_lis = copy(qidnum_lis)
    # 接受id
    query_accept_lis = copy(accept_lis)
    # 打分数
    query_score_lis = copy(qscore_lis)
    # 浏览数
    query_view_lis = copy(qview_lis)
    # 回答数
    query_answer_lis = copy(qanswer_lis)
    # 评论数
    query_comment_lis = copy(qcomment_lis)
    # 喜欢数
    query_favorite_lis = copy(qfavorite_lis)
    # 标题
    query_text_lis = copy(query_lis)
    # 标题上下文
    query_context_lis = copy(qcont_lis)

    ##############################代码的数据###############################
    # 代码帖子id
    code_idnum_lis = copy(cidnum_lis)
    # 打分数
    code_score_lis = copy(cscore_lis)
    # 评论数
    code_comment_lis = copy(ccomment_lis)
    # 代码
    code_text_lis = copy(code_lis)
    # 代码上下文
    code_context_lis = copy(ccont_lis)
    # 最佳排名
    code_rank_lis = copy(crank_lis)

    # 候选代码个数
    mucodes_len = [0]

    # 按键值从小到大排序
    sorted_mucodes_dict={k[0]:k[1] for k in sorted(mucodes_dict.items(),key=lambda mucodes_dict:mucodes_dict[0],reverse=False)}

    count4 = 0
    ##############################添加候选数据###############################
    for id in  sorted_mucodes_dict.keys():
        count4 += 1
        if count4 % 10000 == 0:
            print('处理候选id的已经插入-------------%d条数据' % count4)
        i= 0
        while i < sorted_mucodes_dict[id][1]:
            # 后续索引id = 当前索引 + 前项总长度 + 候选排名
            nid = id + sum(mucodes_len) + i + 1
            # 查询帖子id
            query_idnum_lis.insert(nid,qidnum_lis[id])
            # 接受id
            query_accept_lis.insert(nid,accept_lis[id])
            # 打分数
            query_score_lis.insert(nid,qscore_lis[id])
            # 浏览数
            query_view_lis.insert(nid,qview_lis[id])
            # 回答数
            query_answer_lis.insert(nid,qanswer_lis[id])
            # 评论数
            query_comment_lis.insert(nid,qcomment_lis[id])
            # 喜欢数
            query_favorite_lis.insert(nid,qfavorite_lis[id])
            # 标题
            query_text_lis.insert(nid,query_lis[id])

            #  标题上下文
            query_context_lis.insert(nid,qcont_lis[id])
            # 代码帖子id
            code_idnum_lis.insert(nid,cidnum_lis[id])
            # 打分数
            code_score_lis.insert(nid,cscore_lis[id])
            # 评论数
            code_comment_lis.insert(nid,ccomment_lis[id])
            # 代码
            code_text_lis.insert(nid,sorted_mucodes_dict[id][0][i])
            # 代码上下文
            code_context_lis.insert(nid,ccont_lis[id])
            # 候选排名
            code_rank_lis.insert(nid,i+1)

            i+=1

        # 记录候选个数
        mucodes_len.append(sorted_mucodes_dict[id][1])

    print('回复阶段插入候选总计-------------%d条数据' % count4)

    print('################################### 插入候选执行完毕！###################################')

    # 数据字典
    data_dict = {'ID': query_idnum_lis, 'acceptID': query_accept_lis, 'orgin_query': query_text_lis, 'qcont': query_context_lis,
                 'qview': query_view_lis, 'qanswer': query_answer_lis, 'qfavorite': query_favorite_lis, 'qscore': query_score_lis,
                 'qcomment': query_comment_lis,'keyID':code_idnum_lis,'orgin_code':code_text_lis,'code_rank':code_rank_lis,'ccont':code_context_lis,'cscore':code_score_lis,'ccomment': code_comment_lis}

    # 索引列表
    column_list = ['ID', 'acceptID', 'keyID', 'orgin_query', 'qcont', 'orgin_code','code_rank', 'ccont', 'qview', 'qanswer',
                   'qfavorite', 'qscore', 'qcomment', 'cscore', 'ccomment']
    # 数据保存
    nlpl_data = pd.DataFrame(data_dict, columns=column_list)

    # 删除无效的(包括代码索引不在和首位代码不满足条件)
    nlpl_data = nlpl_data[nlpl_data['orgin_code'] != '-100']  # 索引不在
    nlpl_data = nlpl_data[nlpl_data['orgin_code'] != '-1000']  # 代码不满足条件

    ###################################多选数据处理#########################################
    # 筛选多选code
    rank_first = nlpl_data[nlpl_data['code_rank'] == 0]
    rank_other = nlpl_data[nlpl_data['code_rank'] != 0]
    acce_idx = list(set(rank_first['ID']).intersection(set(rank_other['ID'])))
    mutiple_data = nlpl_data[nlpl_data['ID'].isin(acce_idx)]

    # 重新0-索引
    mutiple_data = mutiple_data.reset_index(drop=True)

    # 单引号置换双引号 便于代码解析
    for name in ['orgin_query', 'qcont', 'orgin_code', 'ccont']:
        mutiple_data[name] = mutiple_data[name].str.replace('\'', '\"')

    # 保存多个候选
    mutiple_data.to_csv(save_folder + 'single_%s_np.csv'%lang_type, encoding='utf-8', sep='|')
    print('多选最终的大小', mutiple_data.shape)  # 格式 csharp:240893,sqlang:259923,javang:214920,python:285335
    print('###################所有多选代码数据库保存成功!#####################')

    ###################################单选数据处理#########################################
    # 删除候选code
    single_data = nlpl_data[nlpl_data['code_rank']==0]

    # 重新0-索引
    single_data = single_data.reset_index(drop=True)

    # 单引号置换双引号 便于代码解析
    for name in ['orgin_query', 'qcont', 'orgin_code', 'ccont']:
        single_data[name] = single_data[name].str.replace('\'', '\"')

    # 保存最佳代码
    single_data.to_csv(save_folder + 'mutiple_%s_np.csv'%lang_type, encoding='utf-8', sep='|')
    print('单选最终的大小', single_data.shape)  # 格式 csharp:171773条,sqlang:211215条,javang:156432,python:150190
    print('###################所有单选代码数据库保存成功!#####################')

    print('-------------------所有查询-代码的数据库保存成功!--------------------')


#参数配置设置
data_folder='./corpus_xml/'

save_folder ='./corpus_csv/'
answer_xml ='answer_2018.xml'
split_num =10000


csharp_name = 'csharp_tag_2018.xml'
sqlang_name = 'sqlang_tag_2018.xml'
javang_name = 'javang_tag_2018.xml'
python_name = 'python_tag_2018.xml'


csharp_type='csharp'
sqlang_type='sqlang'
javang_type='javang'
python_type='python'



if __name__ == '__main__':
    extract_xml(data_folder, csharp_name, sqlang_name, javang_name, python_name, answer_xml)

    # #csharp
    # process_xmldata(data_folder,csharp_type,answer_xml,split_num,save_folder)
    # # sqlang
    # process_xmldata(data_folder,sqlang_type,answer_xml,split_num,save_folder)
    # # javang
    # process_xmldata(data_folder,javang_type,answer_xml,split_num,save_folder)
    # # python
    # process_xmldata(data_folder,python_type,answer_xml,split_num,save_folder)









    









