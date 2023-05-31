import pandas as pd
from svm import Classfier

#分类器
s = Classfier()
s.train("./balanced/pos_train.txt", "./balanced/neg_train.txt")
s.test("./balanced/pos_test.txt", "./balanced/neg_test.txt")


def evalate_npcsv(data_frame,top_k):
    # ---------------------------------------------------------#
    ti_frame = data_frame.sort_values(by='time', ascending=False)
    # 取出前K个作为高质量的语料
    ti_k_data = ti_frame[:top_k]
    # 取出query
    ti_query = ti_k_data['orgin_query'].tolist()
    # 算出标签
    ti_label = [s.filter(i) for i in ti_query]
    # 冗余度
    ti_redun = '%.4f'%(sum(ti_label) / len(ti_label))

    #---------------------------------------------------------#
    tv_frame=data_frame.sort_values(by='qfavorite',ascending=False)
    #取出前K个作为高质量的语料
    tv_k_data=tv_frame[:top_k]
    #取出query
    tv_query=tv_k_data['orgin_query'].tolist()
    #算出标签
    tv_label=[s.filter(i) for i in tv_query]
    #冗余度
    tv_redun='%.4f'%(sum(tv_label)/len(tv_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    df_frame=data_frame.sort_values(by='SN-DFF',ascending=False)
    #取出前K个作为高质量的语料
    df_k_data=df_frame[:top_k]
    #取出query
    df_query=df_k_data['orgin_query'].tolist()
    #算出标签
    df_label=[s.filter(i) for i in df_query]
    #冗余度
    df_redun='%.4f'%(sum(df_label)/len(df_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    hw_frame=data_frame.sort_values(by='SN-WTFF-HW',ascending=False)
    #取出前K个作为高质量的语料
    hw_k_data=hw_frame[:top_k]
    #取出query
    hw_query=hw_k_data['orgin_query'].tolist()
    #算出标签
    hw_label=[s.filter(i) for i in hw_query]
    #冗余度
    hw_redun='%.4f'%(sum(hw_label)/len(hw_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    dw_frame=data_frame.sort_values(by='SN-WTFF-DW',ascending=False)
    #取出前K个作为高质量的语料
    dw_k_data=dw_frame[:top_k]
    #取出query
    dw_query=dw_k_data['orgin_query'].tolist()
    #算出标签
    dw_label=[s.filter(i) for i in dw_query]
    #冗余度
    dw_redun='%.4f'%(sum(dw_label)/len(dw_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    ab_frame=data_frame.sort_values(by='DB-WTFF-AB',ascending=False)
    #取出前K个作为高质量的语料
    ab_k_data=ab_frame[:top_k]
    #取出query
    ab_query=ab_k_data['orgin_query'].tolist()
    #算出标签
    ab_label=[s.filter(i) for i in ab_query]
    #冗余度
    ab_redun='%.4f'%(sum(ab_label)/len(ab_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    tf_frame=data_frame.sort_values(by='TF-IDF',ascending=False)
    #取出前K个作为高质量的语料
    tf_k_data=tf_frame[:top_k]
    #取出query
    tf_query=tf_k_data['orgin_query'].tolist()
    #算出标签
    tf_label = [s.filter(i) for i in tf_query]
    #冗余度
    tf_redun='%.4f'%(sum(tf_label)/len(tf_label))
    #---------------------------------------------------------#

    #---------------------------------------------------------#
    pc_frame=data_frame.sort_values(by='PC-SN-WTFF',ascending=False)
    #取出前K个作为高质量的语料
    pc_k_data=pc_frame[:top_k]
    #取出query
    pc_query=pc_k_data['orgin_query'].tolist()
    #算出标签
    pc_label = [s.filter(i) for i in pc_query]
    #冗余度
    pc_redun='%.4f'%(sum(pc_label)/len(pc_label))
    #---------------------------------------------------------#

    print('MR的冗余度：',ti_redun,'TUV的冗余度：',tv_redun,'DFF的冗余度：',df_redun,'SN-WTFF-HW的冗余度：',hw_redun,'SN-WTFF-DW的冗余度：',dw_redun,'DB-WTFF-αβ的冗余度：',ab_redun,'TF-IDF的冗余度：', tf_redun,'PC-SN-WTFF的冗余度：', pc_redun)

    return [ti_redun,tv_redun,df_redun,hw_redun,dw_redun,ab_redun,tf_redun,pc_redun]


def write_result(test_folder,lang_type,lang_len):
    data_frame = pd.read_csv(test_folder+'%s_np_ex_%s_scored.csv' %(lang_type,lang_len), index_col=0, sep='|', encoding='utf-8')
    evals={}
    for top_k in range(200,lang_len,200):
        evals[top_k]=evalate_npcsv(data_frame,top_k)
        #指定列顺序
    res_frame=pd.DataFrame(evals,index=['MR','TUV','DFF','SN-WTFF-HW','SN-WTFF-DW','DB-WTFF-αβ','TF-IDF','PC-SN-WTFF'])
    res_frame=res_frame.T
    res_frame.to_csv(test_folder+'%s_eval_%s.csv'%(lang_type,lang_len),sep='|',encoding='utf-8')

#20000  200-500   #4000   200-200

test_folder='./corpus_test/'

csharp_type='c#'
csharp_len=4000

sqlang_type='sql'
sqlang_len=4000

if __name__ == '__main__':
    # c#
    write_result(test_folder,csharp_type,csharp_len)
    # sql
    write_result(test_folder,sqlang_type,sqlang_len)
