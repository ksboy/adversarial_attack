import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.activations import softmax
learning_rate = 5e-5
min_learning_rate = 1e-5
batch_size =32
val_batch_size = 512
pred_batch_size = 512

percent_of_epoch = 0.25 * 0.05
num_epochs = 7 //percent_of_epoch
patience = 4
nfolds=5
model_path= "./model"


bert_path = "/home/mhxia/workspace/BDCI/chinese_wwm_ext_L-12_H-768_A-12/"
config_path = bert_path + 'bert_config.json'
checkpoint_path = bert_path + 'bert_model.ckpt'
dict_path = bert_path + 'vocab.txt'


MAX_LEN = 64

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

train= pd.read_csv('./data/train_set.csv')
test=pd.read_csv('./data/dev_set.csv',sep='\t')
train_achievements = train['question1'].values
train_requirements = train['question2'].values
labels = train['label'].values
def label_process(x):
    if x==0:
        return [1,0]
    else:
        return [0,1]
train['label']=train['label'].apply(label_process)
labels_cat=list(train['label'].values)
labels_cat=np.array(labels_cat)
test_achievements = test['question1'].values
test_requirements = test['question2'].values
print(train.shape,test.shape)


def tokenize_data(X1, X2):
    T,T_ = [], []
    for i, _ in enumerate(X1):
        achievements = X1[i]
        requirements = X2[i]
        t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=MAX_LEN)
        T.append(t)
        T_.append(t_)
    T = np.array(T)
    T_ = np.array(T_)
    return T, T_

                
def apply_multiple(input_, layers):
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_
def unchanged_shape(input_shape):
    return input_shape
def substract(input_1, input_2):
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_
def submult(input_1, input_2):
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_
def soft_attention_alignment(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),  ##soft max to each column
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),  ## axis =2 soft max to each row
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
    y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
    return - alpha * y_true * K.log(y_pred) * (1 - y_pred)**gamma\
           - (1 - alpha) * (1 - y_true) * K.log(1 - y_pred) * y_pred**gamma
def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    # for l in bert_model.layers:
    #     l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    tp1 = Lambda(lambda x: K.zeros_like(x))(T1)
    tp2 = Lambda(lambda x: K.zeros_like(x))(T2)
    x1 = bert_model([T1, tp1])
    x2 = bert_model([T2, tp2])
    X1 = Lambda(lambda x: x[:, 0:-1])(x1)
    X2 = Lambda(lambda x: x[:, 0:-1])(x2)

    encode = Bidirectional(LSTM(200, return_sequences=True))
    q1_encoded = encode(X1)
    q2_encoded = encode(X2)
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compose = Bidirectional(GRU(200, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(30, activation='selu')(dense)

    dense = BatchNormalization()(dense)
    output = Dense(2, activation='softmax')(dense)
    model = Model([T1, T2], output)
    model.compile(
        # loss='categorical_crossentropy',
        loss=focal_loss,
        optimizer=Adam(1e-3),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model



skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)

oof_train = np.zeros((len(train), 2), dtype=np.float32)
oof_test = np.zeros((len(test), 2), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    x1 = train_achievements[train_index]
    x2 = train_requirements[train_index]
    x1_token, x2_token = tokenize_data(x1, x2)
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_x2 = train_requirements[valid_index]
    val_x1_token, val_x2_token = tokenize_data(val_x1, val_x2)
    val_y = labels_cat[valid_index]

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint(model_path+"model_%s.w"%fold, monitor='val_accuracy', verbose=1,save_best_only=True, save_weights_only=False, mode='auto')

    model = get_model()
    model.fit(x=[x1_token, x2_token], y=y,
            validation_data= ([val_x1_token, val_x2_token],val_y),
            batch_size=batch_size,
            epochs=num_epochs,
            # steps_per_epoch= (len(x1)+ batch_size -1) // batch_size * percent_of_epoch,
            # validation_steps = (len(val_x1)+ batch_size -1) // batch_size * percent_of_epoch ,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint]
            )
    # model.load_weights('bert{}.w'.format(fold))

    test_x1, test_x2 = tokenize_data(test_achievements, test_requirements)
    oof_test += model.predict((test_x1, test_x2), batch_size=pred_batch_size)
    K.clear_session()
oof_test /= nfolds
test=pd.DataFrame(oof_test)
test.to_csv('test_pred.csv',index=False)
test.head(),test.shape
train=pd.DataFrame(oof_train)
train.to_csv('train_pred.csv',index=False)


pred=pd.read_csv('test_pred.csv').values
pred=pred.argmax(axis=1)
sub=pd.DataFrame()
sub['pred']=list(pred)
sub.to_csv('sub.csv',sep='\t',header=None)
