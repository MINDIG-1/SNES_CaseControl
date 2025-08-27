import pandas as pd
import numpy as np
from pingouin import ancova
from concurrent.futures import ProcessPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import os
from pyS3.utils import *

pxc = px.colors.qualitative



def bootstrap_iteration(df, n_subjects_per_group):
    p_value = None
    cohens_d = None
    try:
        boot_data = pd.concat([group.sample(n=n_subjects_per_group, replace=False) for _, group in df.groupby('group')])

        boot_data['sex'] = boot_data['sex'].map({'M': 0, 'F': 1})

        # ancova_model = ancova(data=boot_data, dv='feature', covar=['age', 'sex'], between='group')
        ancova_model = ancova(data=boot_data, dv='feature', covar=['age', 'sex', 'dataset'], between='group')
        p_value = ancova_model['p-unc'].values[0]
        cohens_d = ancova_model['np2'].values[0]

    except Exception as e:
        print(f"Error during bootstrap iteration: {e}")

    return p_value, cohens_d


def run_bootstrap_analysis(df, n_subjects_per_group, n_bootstrap, n_jobs=-1):
    p_values = []
    cohens_d_values = []

    # Use Parallel to parallelize the bootstrap iterations
    results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_iteration)(df, n_subjects_per_group) for _ in range(n_bootstrap)
    )

    # Collect results
    for p_value, cohens_d in results:
        if p_value is not None:
            p_values.append(p_value)
        if cohens_d is not None:
            cohens_d_values.append(cohens_d)

    return p_values, cohens_d_values

def run_bootstrap_allgrps(df, n_bootstrap, group_list=None, sample_size_list=None, sig_threshold=0.05, samplesize_bin=10):
    if not group_list:
        group_list = ['ASD', 'ADHD', 'ANX', 'LD']
    if not sample_size_list:
        sample_size_list = {'ASD': 604, 'ADHD': 624, 'Learning': 237, 'Anxiety': 323}

    p_values_results = {}
    binary_values_results = {}
    cohens_d_results = {}

    for grp in group_list:
        hc_vs_pathology_df = df[df['group'].isin(['HC', grp])]
        sample_sizes = list(range(10, sample_size_list[grp], samplesize_bin))
        if sample_sizes[-1] != sample_size_list[grp]:
            sample_sizes.append(sample_size_list[grp])

        for n_subjects in sample_sizes:
            print(f"Processing group {grp} with {n_subjects} subjects")
            p_values, cohens_d = run_bootstrap_analysis(hc_vs_pathology_df, n_subjects, n_bootstrap)
            
            binary_values = (np.array(p_values) < sig_threshold).astype(int).tolist()

            p_values_results[(grp, n_subjects)] = p_values
            binary_values_results[(grp, n_subjects)] = binary_values
            cohens_d_results[(grp, n_subjects)] = cohens_d

    return p_values_results, binary_values_results, cohens_d_results


def run_bootstrap_firstnlast(df, n_bootstrap, group_list, sample_size_list, sig_threshold=0.05):

    p_values_results = {}
    binary_values_results = {}
    cohens_d_results = {}

    for grp in group_list:
        hc_vs_pathology_df = df[df['group'].isin(['HC', grp])]
        sample_sizes = [10, sample_size_list[grp]]

        for n_subjects in sample_sizes:
            print(f"Processing group {grp} with {n_subjects} subjects")
            p_values, cohens_d = run_bootstrap_analysis(hc_vs_pathology_df, n_subjects, n_bootstrap)
            
            binary_values = (np.array(p_values) < sig_threshold).astype(int).tolist()

            p_values_results[(grp, n_subjects)] = p_values
            binary_values_results[(grp, n_subjects)] = binary_values
            cohens_d_results[(grp, n_subjects)] = cohens_d

    return p_values_results, binary_values_results, cohens_d_results


def save_to_csv(data_dict, group, file_path, file_prefix):
    columns = []
    values = []
    for key, value in data_dict.items():
        if key[0] == group:
            columns.append(key[1])
            values.append(value)
    df = pd.DataFrame(np.array(values).T, columns=columns)
    df.to_csv(f'{file_path}/{file_prefix}-{group}.csv', index=False)

def over_threshold(data_dict, group, file_path,featureType, fileName, feature_name, nb_bootstraps=1000, m = 95):
    columns = []
    values = []
    for key, value in data_dict.items():
        if key[0] == group:
            columns.append(key[1])
            values.append(value)
    df = pd.DataFrame(np.array(values).T, columns=columns)
    column_sums = df.sum()
    column_sums = column_sums*100/nb_bootstraps
    # if not column_sums[column_sums > m].empty:
    #     sample_size_crossed = column_sums[column_sums > m].index[0]
    #     with open(file_path, 'a') as file:
    #         file.write(f"{feature_name},{group},{sample_size_crossed}\n")

    if not column_sums[column_sums > m].empty:
        sample_size_crossed = column_sums[column_sums > m].index[0]
        with open(file_path, 'a') as file:
            file.write(f"{featureType},{fileName},{feature_name},{group},yes,{sample_size_crossed},{column_sums.iloc[0]:.2f},{column_sums.iloc[-1]:.2f},{column_sums.iloc[-1]-column_sums.iloc[0]:.2f}\n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{featureType},{fileName},{feature_name},{group},no,NaN,{column_sums.iloc[0]:.2f},{column_sums.iloc[-1]:.2f},{column_sums.iloc[-1]-column_sums.iloc[0]:.2f}\n")

def over_threshold_fast(data_dict, group, file_path, feature_name, nb_bootstraps=1000, m = 95):
    columns = []
    values = []
    for key, value in data_dict.items():
        if key[0] == group:
            columns.append(key[1])
            values.append(value)
    df = pd.DataFrame(np.array(values).T, columns=columns)
    column_sums = df.sum()
    column_sums = column_sums*100/nb_bootstraps
    
    if not column_sums[column_sums > m].empty:
        sample_size_crossed = column_sums[column_sums > m].index[0]
        with open(file_path, 'a') as file:
            file.write(f"{feature_name},{group},yes,{column_sums.iloc[0]:.2f},{column_sums.iloc[1]:.2f},{column_sums.iloc[1]-column_sums.iloc[0]:.2f}\n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{feature_name},{group},no,{column_sums.iloc[0]:.2f},{column_sums.iloc[1]:.2f},{column_sums.iloc[1]-column_sums.iloc[0]:.2f}\n")

def heatmap_bootsraps(path_values,col, group_list = ['ASD', 'ADHD', 'Anxiety', 'Learning'], nb_bootstraps = 1000):
    colors = ['#82b2d1', '#749fbb']
    colorscale = [[0, 'white'], [1, colors[0]]]  ##82b2d1 pxc.Pastel[0] pxc.Prism[2]
    file_path = f"IO_files/Outputs/{path_values[0]}/{path_values[1]}/numbers/{path_values[0]}/{col}"
    save_path = f"IO_files/Outputs/{path_values[0]}/{path_values[1]}/figures/{path_values[0]}/{col}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for grp in group_list:
        df = pd.read_csv(f'{file_path}/binval-{grp}.csv')

        heatmap_data = df.T  # Transpose the DataFrame to align columns with y-axis and index with x-axis
        fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.8, 0.2],  # Adjust the width ratio between the scatter plot and histogram
        shared_yaxes=True,  
        horizontal_spacing=0.025)
        
        
        all_counts = (df.sum(axis=0).values/nb_bootstraps)*100
        threshold = 95
        bar_colors = [colors[1] if count <= threshold else colors[0] for count in all_counts]


        # Create the heatmap
        fig.add_trace(go.Heatmap(
            z=heatmap_data.values,  # Values of the heatmap
            x=heatmap_data.columns,  # x-axis values (DataFrame index)
            y=heatmap_data.index,    # y-axis values (DataFrame columns as index after transpose)
            xgap=2,
            ygap = 0,
            colorscale=colorscale ,
            showscale=False    
        ), row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=all_counts,
                # y=y_histo,
                y = df.columns,
                orientation='h',
                width=0.5, 
                marker_color=bar_colors
                # marker = dict(color=pxc.Pastel[0])
            ),row=1, col=2)
        fig.add_shape(
            dict(type="line", x0=threshold, x1=threshold, y0=0, y1=len(df.columns),  # The y-axis span
                line=dict(color=pxc.Pastel[6], width=3, dash="dash")),
                row=1, col=2)
        
        # Update layout for better visualization
        fig.update_layout(
            title= f'{grp}',
            xaxis_title='nb of bootstraps',
            yaxis_title='Sample Size',
            plot_bgcolor='white',
            width=1000,  # Fixed width of the figure
            height=600 
        )
        fig.update_xaxes(title_text='nb of Bootstraps', row=1, col=1)
        fig.update_xaxes(title_text='%', row=1, col=2)
        fig.update_yaxes(tickmode='array', tickvals=df.columns[::3])
        fig.update_xaxes(range=[0, 100], row=1, col=2)



        # Display the figure
        # fig.show()
        file_name = f'Heatmap-{grp}.html' 
        fig.write_html(os.path.join(save_path, file_name))

