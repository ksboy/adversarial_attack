#coding=utf-8
"摘自https://www.biendata.com/forum/view_post_category/718/"
import pandas as pd
import numpy as np
from xml.dom.minidom import parse
from sklearn.model_selection import StratifiedKFold, KFold

def generate_train_data_pair(equ_questions, not_equ_questions):
    a = [x+"\t"+y+"\t"+"0" for x in equ_questions for y in not_equ_questions]
    b = [x+"\t"+y+"\t"+"1" for x in equ_questions for y in equ_questions if x!=y]
    return a+b
    
def parse_train_data(xml_data):
    pair_list = []
    doc = parse(xml_data)
    collection = doc.documentElement
    for i in collection.getElementsByTagName("Questions"):
        # if i.hasAttribute("number"):
        #     print ("Questions number=", i.getAttribute("number"))
        EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
        NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
        equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
        not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
        equ_questions_list, not_equ_questions_list = [], []
        for q in equ_questions:
            try:
                equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        for q in not_equ_questions:
            try:
                not_equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        pair = generate_train_data_pair(equ_questions_list, not_equ_questions_list)
        pair_list.extend(pair)
    print("All pair count=", len(pair_list))
    return pair_list

def xml2csv():
    pair_list = parse_train_data("./data/train_set.xml")
    qid =[]
    question1=[]
    question2=[]
    label=[]
    for i, pair in enumerate(pair_list):
        pair=pair.split('\t')
        qid.append(i)
        question1.append(pair[0])
        question2.append(pair[1])
        label.append(pair[2])
    df=pd.DataFrame()
    df['qid']=qid
    df['question1']=question1
    df['question2']=question2
    df['label']=label
    df.to_csv('./data/train_set.csv',index=False,encoding='utf-8')

def split_data(nfolds=5):
    train= pd.read_csv('./data/train_set.csv')
    label = train['label'].values
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(skf.split(train.values, label)):
        train_fold = train.loc[train_index]
        dev_fold = train.loc[valid_index]
        train_fold.to_csv('./data/%s/train.csv'%fold,index=False,encoding='utf-8')
        dev_fold.to_csv('./data/%s/dev.csv'%fold,index=False,encoding='utf-8')



if __name__ == "__main__":
    xml2csv()
    split_data()


