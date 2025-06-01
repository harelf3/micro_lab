import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

# Function to calculate alpha diversity (Shannon diversity index)
def calculate_alpha_diversity(species_counts):
    proportions = species_counts / np.sum(species_counts)
    proportions = proportions[proportions > 0]  # Avoid log(0)
    return -np.sum(proportions * np.log(proportions))

# Load dataset
data = pd.read_csv('microbiome.csv')  # Replace with your dataset file path
meta_data = pd.read_csv('metadata.csv')  # Replace with your metadata file path

# Filter metadata where PATGROUPFINAL_C is 8
control_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == '8']

# Merge control metadata with data on 'sampleID'
control_data = pd.merge(data, control_meta_data, on='SampleID', how='inner')

# Drop metadata fields from the merged dataset
control_data_columns = [col for col in control_data.columns if col not in meta_data.columns]
filtered_control_data = control_data[control_data_columns].copy()

# Calculate Shannon index (alpha diversity) for each sample in the control group
filtered_control_data.loc[:, 'alpha_diversity'] = filtered_control_data.apply(
    lambda row: calculate_alpha_diversity(row.values), axis=1
)

for i in range(1, 8 ):
    # Filter metadata for the current group
    group_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == str(i)]
    
    # Ensure there are no NaN values in the group metadata before merging
    group_data = pd.merge(data, group_meta_data, on='SampleID', how='inner')
    
    # Drop metadata fields from the merged dataset
    group_data_columns = [col for col in group_data.columns if col not in meta_data.columns]
    filtered_group_data = group_data[group_data_columns].copy()
    
    # Calculate Shannon index (alpha diversity) for each sample in the group
    filtered_group_data.loc[:, 'alpha_diversity'] = filtered_group_data.apply(
        lambda row: calculate_alpha_diversity(row.values), axis=1
    )
    
    print(f"Group {i} alpha diversity mean: {filtered_group_data['alpha_diversity'].mean()}")
    print(f"Control group alpha diversity mean: {filtered_control_data['alpha_diversity'].mean()}")
    
    # Perform t-test between the control group and the current group
    t_stat, p_value = ttest_ind(
        filtered_control_data['alpha_diversity'], 
        filtered_group_data['alpha_diversity'], 
        equal_var=False
    )
    
    print(f"Group {i} vs Control: t-statistic = {t_stat}, p-value = {p_value}")
    
    # Box plot comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [filtered_control_data['alpha_diversity'], filtered_group_data['alpha_diversity']],
        labels=['Control (8)', f'Group {i}']
    )
    plt.title(f'Alpha Diversity Comparison: Control vs Group {i}')
    plt.xlabel('Group')
    plt.ylabel('Alpha Diversity')
    plt.show()
# # Loop through each PATGROUPFINAL_C value from 1 to 7
# for i in range(1, 8):
#     # Filter metadata for the current group
#     group_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == i]
    
#     # Ensure there are no NaN values in the group metadata before merging
#     group_data = pd.merge(data, group_meta_data, on='SampleID', how='inner')
    
#     # Drop metadata fields from the merged dataset
#     group_data_columns = [col for col in group_data.columns if col not in meta_data.columns]
#     filtered_group_data = group_data[group_data_columns]
    
#     # Calculate Shannon index (alpha diversity) for each sample in the group
#     filtered_group_data['alpha_diversity'] = filtered_group_data.apply(
#         lambda row: calculate_alpha_diversity(row.values), axis=1
#     )
#     print(f"Group {i} alpha diversity mean: {filtered_group_data['alpha_diversity'].mean()}")
#     print(f"Control group alpha diversity mean: {filtered_control_data['alpha_diversity'].mean()}")
#     print(f"Group {i} sample size: {filtered_group_data.head(10)}")
#     # Perform t-test between the control group and the current group
#     t_stat, p_value = ttest_ind(
#         filtered_control_data['alpha_diversity'], 
#         filtered_group_data['alpha_diversity'], 
#         equal_var=False
#     )
#     print(f"Group {i} vs Control: t-statistic = {t_stat}, p-value = {p_value}")
    
#     # Box plot comparison
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(
#         [filtered_control_data['alpha_diversity'], filtered_group_data['alpha_diversity']],
#         labels=['Control (8)', f'Group {i}']
#     )
#     plt.title(f'Alpha Diversity Comparison: Control vs Group {i}')
#     plt.xlabel('Group')
#     plt.ylabel('Alpha Diversity')
#     plt.show()
# # Group by sickness and calculate mean alpha diversity
# sickness_alpha_diversity = data.groupby('sickness')['alpha_diversity'].mean()

# # Separate sick and control groups
# data['species_counts'] = data['species_counts'].apply(lambda x: len(eval(x)))
# sick_group = data[data['group'] == 'sick']['alpha_diversity']
# control_group = data[data['group'] == 'control']['alpha_diversity']

# # Count the number of species in each sample
# data['species_count'] = data['species_counts'].apply(lambda x: len(eval(x)))

# # Perform t-test
# t_stat, p_value = ttest_ind(sick_group, control_group, equal_var=False)
# print(f"T-test results: t-statistic = {t_stat}, p-value = {p_value}")

# # Box plot comparison
# plt.figure(figsize=(8, 6))
# data.boxplot(column='alpha_diversity', by='group', grid=False)
# plt.title('Alpha Diversity Comparison')
# plt.suptitle('')
# plt.xlabel('Group')
# plt.ylabel('Alpha Diversity')
# plt.show()