import pandas as pd
import numpy as np
from pingouin import ancova
from statsmodels.stats.multitest import fdrcorrection
import os
from utils import *
import time


BUCKET = 'nm-psy'
if __name__ == '__main__':
    t1 = time.time()
    column_names = ['featureType', 'bucket', 'fileName','featureName', 'group']
    feat_list_df = pd.read_csv('IO_files/Outputs/feature_list.txt', delimiter=',', header=None, names=column_names)

    column_names = ['featureType', 'bucket', 'fileName','featureName', 'group', 'threshold']
    if os.path.exists('IO_files/Outputs/Crossed.txt'):
        prev_done = pd.read_csv('IO_files/Outputs/Crossed.txt', delimiter=',', header=None, names=column_names)
        done_feats = feat_list_df.merge(prev_done[['featureName', 'group']], on = ['featureName', 'group'], how = 'left', indicator = True)
        feat_list_df = done_feats[done_feats['_merge'] == 'left_only'].drop(columns = ['_merge'])

    for row_i, row_df  in feat_list_df.iterrows():
        path_values = list(row_df[['featureType','bucket','fileName','featureName']].values)
        print(row_df['featureName'], row_df['group'])

        nb_bootstraps=100
        samplesize_bin=100

        df = pd.read_csv(f"IO_files/Inputs/{path_values[0]}/{path_values[1]}/{path_values[2]}")

        df = df[(df['age']>=5) & (df['age']<=18)]
        group_list = ['ASD', 'ADHD', 'Anxiety', 'Learning']
        HC_size = len(df[df['group']=='HC'])
        sample_size_list = {'ASD': min(len(df[df['group']=='ASD']),HC_size), 'ADHD': min(len(df[df['group']=='ADHD']),HC_size), 'LD': min(len(df[df['group']=='LD']),HC_size), 'ANX': min(len(df[df['group']=='ANX']),HC_size)}

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dataset_col_pos = 5
        df['dataset_names'] = df['dataset']
        df['dataset'] = pd.Categorical(df['dataset']).codes
        df.insert(dataset_col_pos, 'dataset_names', df.pop('dataset_names'))

        df_sub = df[['global_id','age','sex','group','dataset', row_df['featureName']]].rename(columns={row_df['featureName']: 'feature'})
        pval, bin_val, cohenval = run_bootstrap_allgrps(df_sub, nb_bootstraps, [row_df['group']], sample_size_list, samplesize_bin=samplesize_bin)

        file_path = f"IO_files/Outputs/{path_values[0]}/{path_values[1]}/numbers/{path_values[0]}/{row_df['featureName']}"
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        save_to_csv(pval, row_df['group'],file_path, 'pval')
        save_to_csv(bin_val, row_df['group'],file_path, 'binval')
        save_to_csv(cohenval, row_df['group'],file_path, 'es')
        crossed_file_path = 'IO_files/Outputs/Crossed.txt'
        # over_threshold(data_dict, group, file_path,featureType, fileName, feature_name, nb_bootstraps=1000, m = 95):
        over_threshold(bin_val, row_df['group'],crossed_file_path,row_df['featureType'], row_df['fileName'], row_df['featureName'], nb_bootstraps)
        # over_threshold(bin_val,row_df['group'],crossed_file_path,f"{path_values[0]},{path_values[1]},{path_values[2]},{row_df['featureName']}")
        heatmap_bootsraps(path_values, row_df['featureName'], [row_df['group']])
        print(f"Runtime: {time.time()-t1:.2f}s | {(time.time()-t1)/3600:.2f} hrs")


    print(f"Runtime: {time.time()-t1:.2f}s | {(time.time()-t1)/3600:.2f} hrs")