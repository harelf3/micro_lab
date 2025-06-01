import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform # For Bray-Curtis distance calculation
from sklearn.manifold import MDS # For PCoA
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced plotting

# Function to calculate alpha diversity (Shannon diversity index) - provided by user
def calculate_alpha_diversity(species_counts):
    # Ensure species_counts is a numpy array and handle potential non-numeric entries if any
    species_counts = np.asarray(species_counts, dtype=float)
    # Avoid log(0) for species with 0 counts by filtering them out
    proportions = species_counts / np.sum(species_counts)
    proportions = proportions[proportions > 0]
    
    # Handle cases where all proportions are zero (e.g., sample with no reads)
    if len(proportions) == 0:
        return 0.0 # Shannon diversity is 0 if there are no species or no reads
    
    return -np.sum(proportions * np.log(proportions))

# Load dataset
data = pd.read_csv('microbiome_relative_abundance.csv')  # Replace with your dataset file path
meta_data = pd.read_csv('metadata.csv')  # Replace with your metadata file path

# --- PCoA Analysis for Beta Diversity (Bray-Curtis Dissimilarity) ---
print("--- Starting Beta Diversity Analysis (Bray-Curtis and PCoA) ---")

# 1. Prepare the species abundance matrix for Bray-Curtis calculation
# We assume that 'data' DataFrame contains 'SampleID' and then only species abundance columns.
# Identify columns that represent species abundances (all columns except 'SampleID').
species_cols = [col for col in data.columns if col != 'SampleID']

# Extract the abundance data and ensure it's numeric (float type is robust)
abundance_matrix = data[species_cols].astype(float).values

# Check for samples with zero total counts, which can cause issues with Bray-Curtis
row_sums = np.sum(abundance_matrix, axis=1)
if np.any(row_sums == 0):
    print("Warning: Some samples have zero total counts. These samples will have NaN/Inf Bray-Curtis distances to others if those also have zero counts, or 1.0 distance if other samples have counts. Consider filtering them out if this is not desired.")
    # For now, we proceed, but be aware of potential issues in distance calculations.
    # If you want to filter them, uncomment and adapt the following:
    # valid_samples_mask = row_sums > 0
    # abundance_matrix = abundance_matrix[valid_samples_mask]
    # pcoa_sample_ids = data.loc[valid_samples_mask, 'SampleID'].reset_index(drop=True)
else:
    pcoa_sample_ids = data['SampleID'] # Keep original SampleIDs if no filtering

# 2. Calculate Bray-Curtis distance matrix
# pdist computes pairwise distances between observations (rows) in the abundance matrix.
# The output is a condensed distance matrix (a 1D array).
bray_curtis_distances = pdist(abundance_matrix, metric='braycurtis')

# Convert the condensed distance matrix to a squareform (symmetric) matrix
bray_curtis_matrix = squareform(bray_curtis_distances)

# 3. Perform PCoA using MDS (Multidimensional Scaling)
# PCoA is a form of MDS performed on a distance matrix.
# 'dissimilarity='precomputed'' tells MDS to interpret the input as a distance matrix.
# 'n_components=2' projects data into 2 dimensions for a 2D plot.
# 'random_state' for reproducibility of results.
# 'n_init=1' avoids multiple random initializations when using 'precomputed' distances, making it faster and deterministic.
# 'max_iter' helps prevent very long runtimes for large datasets by limiting iterations.
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=500)
pcoa_results = mds.fit_transform(bray_curtis_matrix)

# Convert PCoA results (the principal coordinates) to a DataFrame for easier plotting
pcoa_df = pd.DataFrame(pcoa_results, columns=['PCo1', 'PCo2'])

# Add SampleID back to pcoa_df so we can merge with metadata
pcoa_df['SampleID'] = pcoa_sample_ids # Use the SampleIDs corresponding to the abundance_matrix

# 4. Merge PCoA results with metadata for plotting
# This allows us to color the points by 'PATGROUPFINAL_C' (patient group).
pcoa_df_merged = pd.merge(pcoa_df, meta_data[['SampleID', 'PATGROUPFINAL_C']], on='SampleID', how='inner')

# Ensure 'PATGROUPFINAL_C' is treated as categorical for plotting (convert to string)
pcoa_df_merged['PATGROUPFINAL_C'] = pcoa_df_merged['PATGROUPFINAL_C'].astype(str)

# 5. Visualize PCoA results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=pcoa_df_merged,
    x='PCo1',
    y='PCo2',
    hue='PATGROUPFINAL_C', # Color points by patient group
    palette='tab10',       # A distinct color palette for categorical data
    s=100,                 # Marker size
    alpha=0.8,             # Transparency of points
    edgecolor='w',         # Add white edge for better visibility of points
    linewidth=0.5
)
plt.title('Principal Coordinate Analysis (PCoA) of Bray-Curtis Dissimilarity')
plt.xlabel('Principal Coordinate 1 (PCo1)')
plt.ylabel('Principal Coordinate 2 (PCo2)')
plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
plt.legend(title='Patient Group', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside
plt.tight_layout() # Adjust plot to prevent labels/legend from overlapping
plt.show()

print("--- PCoA Analysis Complete ---")

# --- Original Alpha Diversity Analysis (Shannon Diversity Index) ---
print("\n--- Starting Alpha Diversity Analysis (Shannon Index) ---")

# Filter metadata where PATGROUPFINAL_C is 8 (Control Group)
control_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == '8']

# Merge control metadata with data on 'SampleID'
control_data = pd.merge(data, control_meta_data, on='SampleID', how='inner')

# Isolate species count columns for alpha diversity calculation
# We need to exclude 'SampleID' and any potential metadata columns that might have merged in.
# A robust way is to select only the original species columns from the 'data' DataFrame.
control_data_for_alpha = control_data[[col for col in data.columns if col != 'SampleID']].copy()

# Calculate Shannon index (alpha diversity) for each sample in the control group
control_data_for_alpha.loc[:, 'alpha_diversity'] = control_data_for_alpha.apply(
    lambda row: calculate_alpha_diversity(row.values), axis=1
)
# Create a DataFrame to hold SampleID and its calculated alpha diversity for easier merging/use
filtered_control_data_with_alpha = pd.DataFrame({
    'SampleID': control_data['SampleID'],
    'alpha_diversity': control_data_for_alpha['alpha_diversity']
})


all_unique_patient_groups = sorted(meta_data['PATGROUPFINAL_C'].unique().tolist())

# Loop through each unique patient group (e.g., '1', '2', ..., '8')
for current_group_id_str in all_unique_patient_groups:
    # Filter metadata for the current group
    # Ensure comparison is with string representation as PATGROUPFINAL_C might be string type in metadata
    group_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == str(current_group_id_str)]
    
    # Merge current group metadata with data on 'SampleID'
    group_data = pd.merge(data, group_meta_data, on='SampleID', how='inner')
    
    # Isolate species count columns for alpha diversity calculation for the current group
    group_data_for_alpha = group_data[[col for col in data.columns if col != 'SampleID']].copy()
    
    # Calculate Shannon index (alpha diversity) for each sample in the group
    group_data_for_alpha.loc[:, 'alpha_diversity'] = group_data_for_alpha.apply(
        lambda row: calculate_alpha_diversity(row.values), axis=1
    )
    # Create a DataFrame to hold SampleID and its calculated alpha diversity for easier merging/use
    filtered_group_data_with_alpha = pd.DataFrame({
        'SampleID': group_data['SampleID'],
        'alpha_diversity': group_data_for_alpha['alpha_diversity']
    })
    
    print(f"\nGroup {current_group_id_str} alpha diversity mean: {filtered_group_data_with_alpha['alpha_diversity'].mean():.4f}")
    print(f"Control group alpha diversity mean: {filtered_control_data_with_alpha['alpha_diversity'].mean():.4f}")
    
    # Perform t-test between the control group and the current group
    # Using Welch's t-test (equal_var=False) which does not assume equal population variances
    t_stat, p_value = ttest_ind(
        filtered_control_data_with_alpha['alpha_diversity'], 
        filtered_group_data_with_alpha['alpha_diversity'], 
        equal_var=False
    )
    
    print(f"Group {current_group_id_str} vs Control: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    # Box plot comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [filtered_control_data_with_alpha['alpha_diversity'], filtered_group_data_with_alpha['alpha_diversity']],
        labels=['Control (8)', f'Group {current_group_id_str}']
    )
    plt.title(f'Alpha Diversity Comparison: Control vs Group {current_group_id_str}')
    plt.xlabel('Group')
    plt.ylabel('Alpha Diversity (Shannon Index)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

print("--- Alpha Diversity Analysis Complete ---")