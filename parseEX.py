#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from bs4 import BeautifulSoup


# --------------------------解析XML文件的标签-----------------------------#
#滤除非法字符(不滤除后续解析出错)
def filter_inva_char(line):
    #去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|#|/|%|_|,|\'|:|=|>|<|\"|;|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

#解析属性
def get_tag_param(rowTag,tag):
    try:
        output = rowTag[tag]
        return output
    except KeyError:
        return -10000

#存储query的数据
def extract_idnl(data_folder,lang_type,save_folder,test_nlpart,save_nlk):
    #帖子id，接受id
    idnum_lis,accept_lis=[],[]
    #打分数
    score_lis=[]
    #浏览数、回答数、评论数、喜欢数
    view_lis,answer_lis,comment_lis,favorite_lis=[],[],[],[]
    #标题
    query_lis=[]
    #contxet
    qcont_lis=[]
    #时间戳
    time_lis=[]
    #计数
    count1=0
    count2=0
    with open(data_folder+'%s_tag_2018.xml'%lang_type,'r',encoding='utf-8') as f:
        for line in f:
            count1+=1
            if count1 %10000==0: #总计
                print('查询阶段已经处理-%d条数据'%count1)
            rowTag=BeautifulSoup(line,'lxml').row #row标签
            #标题
            query=get_tag_param(rowTag,'title')
            if query!=-10000:
                accept=get_tag_param(rowTag,'acceptedanswerid')
                if accept!=-10000 and lang_type in get_tag_param(rowTag, "tags") :  # 存在
                    count2+=1
                    if count2%10000==0:
                        print('查询阶段已经保存-------------%d条数据'%count2)
                    if test_nlpart and  count2%(save_nlk+1)==0:
                        break
                    #查询query
                    query_lis.append(str(query))
                    #接受id
                    accept_lis.append(int(accept))
                    #帖子id
                    idnum=get_tag_param(rowTag,'id')
                    idnum_lis.append(int(idnum))
                    #打分数
                    scorenum=get_tag_param(rowTag,'score')
                    score_lis.append(int(scorenum))
                    #浏览数
                    viewnum=get_tag_param(rowTag,'viewcount')
                    view_lis.append(int(viewnum))
                    #回答数
                    answernum=get_tag_param(rowTag,'answercount')
                    answer_lis.append(int(answernum))
                    #评论数
                    comment=get_tag_param(rowTag,'commentcount')
                    comment_lis.append(int(comment))
                    #喜欢数
                    favorite=get_tag_param(rowTag,'favoritecount')
                    favorite_lis.append(int(favorite))
                    #时间戳
                    time=get_tag_param(rowTag,'creationdate')
                    time_lis.append(str(time))
                    #body属性
                    body=get_tag_param(rowTag,'body')
                    soup=BeautifulSoup(body,'lxml')
                    # context
                    qcont = soup.findAll(text=True)
                    # 去除噪音
                    qcont = ' '.join([filter_inva_char(i) for i in qcont])
                    qcont = re.sub('\n+', '\n', re.sub(' +', ' ', qcont)).strip('\n').replace('\n', '\\n')
                    qcont_lis.append(qcont)
        print('总计查询了%d条数据'%count1) #c#:1141067条,sql:1947440条 #c#:55496条，sql:102440条
        print('总计保存了%d条数据'%count2) #c#:642662条,sql:602405条   #c#:40001条，sql:40001条

    query_dict={'ID':idnum_lis,'acceptID':accept_lis,'orgin_query':query_lis,'qcont':qcont_lis,'qview':view_lis,'qanswer': answer_lis,'qfavorite':favorite_lis,'qscore':score_lis,'qcomment': comment_lis,'time': time_lis}
    query_data=pd.DataFrame(query_dict,columns=['ID','acceptID','orgin_query','qcont','qview','qanswer','qfavorite','qscore','qcomment','time'])
    #重新0-索引
    query_data=query_data.reset_index(drop=True)

    if test_nlpart:
        #保存数据
        query_data.to_csv(save_folder+'%s_nl_ex_%s_origin.csv'%(lang_type,save_nlk),encoding='utf-8',sep='|')
    else:
        # 保存数据
        query_data.to_csv(save_folder+'%s_nl_ex_all_origin.csv'%lang_type, encoding='utf-8', sep='|')
    print('查询阶段保存成功！')


#获取code中数据（满足条件）
def get_code_text(codeTag):
    if len(codeTag) == 1:
        code = codeTag[0].get_text().strip()
        #代码片段6-1000字符,过滤字符长度
        if (len(code) >=10 and len(code) <= 1000):
            #过滤代码的关键片段
            if code[0] == '<' or code[0] == '=' or code[0] == '@'  or code[0] == '$' or \
                code[0:7].lower() == 'select' or code[0:7].lower() == 'update' or code[0:6].lower() == 'alter' or \
                code[0:2].lower() == 'c:' or code[0:4].lower() == 'http' or code[0:4].lower() == 'hkey' or \
                re.match(r'^[a-zA-Z0-9_]*$', code) is not None:  #最后一个是相关答案
                return  -10000
            else:
                code = re.sub('\n+', '\n', re.sub(' +', ' ', filter_inva_char(code))).strip('\n').replace('\n','\\n')
                return  code
        else:
            return -10000
    else:
        return -10000

#索引answer属性(索引acceptid)
def extract_idpl(save_folder,lang_type,test_nlpart,save_nlk,data_folder,answer_xml,split_num,part_plk):

    if test_nlpart:
        reader=pd.read_csv(save_folder+'%s_nl_ex_%s_origin.csv'%(lang_type,save_nlk),index_col=0,encoding='UTF-8',sep='|')
    else:
        reader=pd.read_csv(save_folder+'%s_nl_ex_all_origin.csv'%lang_type, index_col=0, encoding='UTF-8',sep='|')

    # accept接受的标签
    accept_lis=reader['acceptID'].tolist()

    ###############################################找寻相关的XML文件#########################################

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [accept_lis[i:i+split_num] for i in range(0, len(accept_lis), split_num)]

    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_part%d.xml'%(lang_type,(i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str,data_folder+answer_xml, data_folder+epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans.xml'%lang_type)
    # 执行命令
    os.popen(ping_cmd).read()     #得到change_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_part*.xml'%lang_type
    os.popen(remove_cmd).read()
    print('XML部分文件删除完毕!')

    ####################################解析具有全部accept_lis索引的change_xml文件#######################################

    index_num=len(accept_lis)  #索引的长度
    print('索引的长度:\t',index_num)

    #帖子id
    idnum_lis=[]
    #打分数
    score_lis=[]
    #评论数
    comment_lis=[]
    #代码
    code_lis=[]
    #context
    ccont_lis=[]

    #计数
    count1=0
    count2=0

    with open(data_folder+'%s_ans.xml'%lang_type,'r',encoding='utf-8') as f:
        for line in f:
            count1+=1
            if count1 %1000==0:
                print('回复阶段已经处理%d条数据'%count1)
            sta=line.find("\"")
            end=line.find("\"", sta + 1)
            qid=int(line[(sta + 1):end])
            if qid in accept_lis: #qid不会重复
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性
                body = get_tag_param(rowTag, 'body')
                soup = BeautifulSoup(body, 'lxml')
                # 代码code
                codeTag = soup.find_all('code')  #code标签
                code = get_code_text(codeTag)    #code文本
                # 保证代码存在
                if code != -10000:
                    count2 += 1
                    if count2 % 100 == 0:
                        print('回复阶段已经保存-------------%d条数据' % count2)
                    if count2 % (part_k+1) == 0:
                        break
                    code_lis.append(code)
                    # 帖子id
                    idnum = get_tag_param(rowTag, 'id')
                    idnum_lis.append(int(idnum))
                    # 打分数
                    scorenum = get_tag_param(rowTag, 'score')
                    score_lis.append(int(scorenum))
                    # 评论数
                    comment = get_tag_param(rowTag, 'commentcount')
                    comment_lis.append(int(comment))
                    # context
                    ccont = soup.findAll(text=True)
                    # 去除噪音
                    ccont = ' '.join([filter_inva_char(i) for i in ccont])
                    ccont = re.sub('\n+', '\n', re.sub(' +', ' ', ccont)).strip('\n').replace('\n','\\n')
                    ccont_lis.append(ccont)

        print('回复阶段总计查询了%d条数据'%count1)   #c#:97382条,sql:78928条条   #c#:577129条，sql:751871条
        print('回复阶段总计保存了%d条数据'%count2)   #c#:20001条,sql:20001条          #c#:4000条，sql:4000条

    code_dict={'keyID':idnum_lis,'orgin_code':code_lis,'ccont':ccont_lis,'cscore':score_lis,'ccomment': comment_lis}
    code_data=pd.DataFrame(code_dict,columns=['keyID','orgin_code','ccont','cscore','ccomment'])
    #重新0-索引
    code_data=code_data.reset_index(drop=True)
    #保存
    code_data.to_csv(save_folder+'%s_pl_ex_%s_origin.csv'%(lang_type,part_plk),encoding='utf-8',sep='|')
    print('回复阶段保存成功！')


def concat_nlpl(save_folder,lang_type,test_nlpart,save_nlk,part_plk):
    #code数据
    code_data=pd.read_csv(save_folder+'%s_pl_ex_%s_origin.csv'%(lang_type,part_plk),index_col=0, encoding='utf-8', sep='|')
    #抽取代码
    cpart_id=code_data['keyID'].tolist()
    #query数据
    if test_nlpart:
        query_data=pd.read_csv(save_folder+'%s_nl_ex_%s_origin.csv'%(lang_type,save_nlk),index_col=0,encoding='UTF-8',sep='|')
    else:
        query_data=pd.read_csv(save_folder+'%s_nl_ex_all_origin.csv'%lang_type, index_col=0, encoding='UTF-8',sep='|')
    #过滤操作 列-索引
    query_data=query_data.set_index('acceptID').reindex(index=cpart_id)
    #索引-列
    query_data['acceptID']=query_data.index
    # 重建0-索引
    query_data = query_data.reset_index(drop=True)
    #拼接DataFrame
    result_data=pd.concat([query_data,code_data],axis=1)
    #索引列表
    column_list=['ID','acceptID','keyID','orgin_query','qcont','orgin_code','ccont','qview','qanswer','qfavorite','qscore','qcomment','cscore','ccomment','time']
    #重列-索引
    result_data=result_data.reindex(columns=column_list)
    #重建0-索引
    result_data=result_data.reset_index(drop=True)
    result_data.to_csv(save_folder+'%s_np_ex_%s_origin.csv'%(lang_type,part_plk),encoding='utf-8',sep='|')


#参数配置
data_folder = './corpus_xml/'
#CSV保存数据
save_folder ='./corpus_test/'
#XML标签
answer_xml='answer_2018.xml'
split_num =10000

#语言配置  csharp/sql/java/python
lang_type ='c#'

# #全部抽取
# test_part=False
# #参数无效
# save_k=0
# #部分大小
# part_k=20000

#部分抽取
test_part=True
#抽取大小
save_k=40000
# 部分大小
part_k=4000


if __name__ == '__main__':
    extract_idnl(data_folder,lang_type,save_folder,test_part,save_k)
    extract_idpl(save_folder,lang_type,test_part,save_k,data_folder,answer_xml,split_num,part_k)
    concat_nlpl(save_folder,lang_type,test_part,save_k,part_k)
