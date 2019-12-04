# # coding=utf-8
# # Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# # Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """ GLUE processors and helpers """

# import logging
# import os
# import csv, sys

# from transformers import DataProcessor,InputExample, InputFeatures
# from transformers import is_tf_available
# from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score

# def simple_accuracy(preds, labels):
#     return (preds == labels).mean()

# def acc_and_macro_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
#     f1_pos= f1_score(y_true=labels, y_pred=preds,  pos_label= 1)
#     recall_pos = recall_score(y_true=labels, y_pred=preds,  pos_label= 1)
#     precision_pos = precision_score(y_true=labels, y_pred=preds,  pos_label= 1)
#     f1_neg = f1_score(y_true=labels, y_pred=preds,  pos_label= 0)
#     recall_neg = recall_score(y_true=labels, y_pred=preds,  pos_label= 0)
#     precision_neg = precision_score(y_true=labels, y_pred=preds,  pos_label= 0)

#     return {
#         "metric":f1,
#         "acc": acc,
#         "f1_pos":f1_pos,
#         "recall_pos":recall_pos,
#         "precision_pos":precision_pos,
#         "f1_neg":f1_neg,
#         "recall_neg":recall_neg,
#         "precision_neg":precision_neg,
#         "f1": f1
#     }

# def acc_and_binary_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds, pos_label= 1)
#     return {
#         "metric":f1,
#         "acc": acc,
#         "f1": f1
#     }

# def glue_compute_metrics(task_name, preds, labels):
#     assert len(preds) == len(labels)
#     if task_name == "aa":
#         return acc_and_macro_f1(preds, labels)
#     else:
#         raise KeyError(task_name)

        


# if is_tf_available():
#     import tensorflow as tf

# logger = logging.getLogger(__name__)



# class AAProcessor(DataProcessor):
#     """Processor for the WNLI data set (GLUE version)."""

#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(tensor_dict['idx'].numpy(),
#                             tensor_dict['sentence1'].numpy().decode('utf-8'),
#                             tensor_dict['sentence2'].numpy().decode('utf-8'),
#                             str(tensor_dict['label'].numpy()))

#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_csv(os.path.join(data_dir, "train.csv")), "train")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")
    
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training, dev and test sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = self.get_labels()[0] if set_type=='test' else line[-1]

#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
    
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training, dev and test sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = self.get_labels()[0] if set_type=='test' else line[-1]

#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
    
#     @classmethod
#     def _read_csv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r", encoding="utf-8-sig") as f:
#             reader = csv.reader(f, quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 if sys.version_info[0] == 2:
#                     line = list(unicode(cell, 'utf-8') for cell in line)
#                 lines.append(line)
#             return lines
    

# glue_tasks_num_labels = {
#     "aa": 2,
# }

# glue_processors = {
#     "aa": AAProcessor,
# }

# glue_output_modes = {
#     "aa": "classification",
# }
