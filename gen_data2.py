#coding=utf-8
"摘自https://www.biendata.com/forum/view_post_category/718/"
import os
import pandas as pd
import numpy as np
from xml.dom.minidom import parse
from sklearn.model_selection import StratifiedKFold, KFold
from random import shuffle

def generate_train_data_pair(train_group):
    train_pair= []
    for example in train_group:
        equ_questions, not_equ_questions= example[0], example[1]
        a = [x+"\t"+y+"\t"+"0" for x in equ_questions for y in not_equ_questions]
        b = [x+"\t"+y+"\t"+"1" for x in equ_questions for y in equ_questions if x!=y]
        train_pair.extend(a+b)
    return train_pair
def parse_train_data(xml_data):
    group_list = []
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
        group_list.append([equ_questions_list, not_equ_questions_list])

    print("All group count=", len(group_list))
    return group_list


def xml2csv(pair_list, out_path):
    shuffle(pair_list)
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
    df.to_csv(out_path,index=False,encoding='utf-8')

def split_data(nfolds=5):
    group_list = parse_train_data("./data/train_set.xml")
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(kf.split(group_list)):
        train_fold = [group_list[i] for i in train_index]
        dev_fold = [group_list[i] for i in valid_index]
        output_dir = "./data/group_wise/%s"%fold
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        xml2csv(generate_train_data_pair(train_fold), os.path.join(output_dir, 'train.csv'))
        xml2csv(generate_train_data_pair(dev_fold), os.path.join(output_dir, 'dev.csv'))


if __name__ == "__main__":
    split_data()


