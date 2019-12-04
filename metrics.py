# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_macro_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_pos= f1_score(y_true=labels, y_pred=preds,  pos_label= 1)
    recall_pos = recall_score(y_true=labels, y_pred=preds,  pos_label= 1)
    precision_pos = precision_score(y_true=labels, y_pred=preds,  pos_label= 1)
    f1_neg = f1_score(y_true=labels, y_pred=preds,  pos_label= 0)
    recall_neg = recall_score(y_true=labels, y_pred=preds,  pos_label= 0)
    precision_neg = precision_score(y_true=labels, y_pred=preds,  pos_label= 0)

    return {
        "metric":f1,
        "acc": acc,
        "f1_pos":f1_pos,
        "recall_pos":recall_pos,
        "precision_pos":precision_pos,
        "f1_neg":f1_neg,
        "recall_neg":recall_neg,
        "precision_neg":precision_neg,
        "f1": f1
    }

def acc_and_binary_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, pos_label= 1)
    return {
        "metric":f1,
        "acc": acc,
        "f1": f1
    }

def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "aa":
        return acc_and_macro_f1(preds, labels)
    else:
        raise KeyError(task_name)

        