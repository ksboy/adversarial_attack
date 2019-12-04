
import pickle
import numpy as np
import pandas as pd
pred_file = "/home/mhxia/whou/workspace/my_models/adversarial_attack/0/checkpoint-best/predict_results.txt"
preds=pickle.load(open(pred_file,'rb'))
preds = np.argmax(preds, axis=1)
result_file="./result.csv"
outf = open(result_file,'w')
df = pd.DataFrame({'pred':preds})
df.to_csv(result_file, header= False, index=True, sep='\t')

