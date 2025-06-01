import math
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('microbiome.csv')  # Replace with your dataset file path
meta_data = pd.read_csv('metadata.csv')  # Replace with your metadata file path
feature_cols = [col for col in data.columns if col != 'SampleID']
aggregated_rows_dict = {}

all_unique_patient_groups = sorted(meta_data['PATGROUPFINAL_C'].unique().tolist())


# Filter metadata for the current group
# Ensure comparison is with string representation as PATGROUPFINAL_C might be string type in metadata
group_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == str(8)]

# Merge current group metadata with data on 'SampleID'
group_data = pd.merge(data, group_meta_data, on='SampleID', how='inner')

summed_bacteria_for_group = group_data[feature_cols].sum(axis=0)

# Store this summed Series in our dictionary, using the group ID as the key.
# Each entry in the dictionary will become one row in the final DataFrame.
aggregated_rows_dict[str(8)] = summed_bacteria_for_group
relative_abundance_for_control = summed_bacteria_for_group / summed_bacteria_for_group.sum()

descriptive_comparison_results = {}


# Loop through each unique patient group (e.g., '1', '2', ..., '8')
for current_group_id_str in all_unique_patient_groups:
    # Filter metadata for the current group
    # Ensure comparison is with string representation as PATGROUPFINAL_C might be string type in metadata
    group_meta_data = meta_data[meta_data['PATGROUPFINAL_C'] == str(current_group_id_str)]
    
    # Merge current group metadata with data on 'SampleID'
    group_data = pd.merge(data, group_meta_data, on='SampleID', how='inner')
    
    summed_bacteria_for_group = group_data[feature_cols].sum(axis=0)
    
    # Store this summed Series in our dictionary, using the group ID as the key.
    # Each entry in the dictionary will become one row in the final DataFrame.
    aggregated_rows_dict[str(current_group_id_str)] = summed_bacteria_for_group
    relative_abundance_for_group = summed_bacteria_for_group / summed_bacteria_for_group.sum()
    epsilon = 1e-9 # A very small number
    
    # Calculate fold change for each feature
    # Fold Change = (Abundance in Group) / (Abundance in Control)
    # log2(Fold Change) is often preferred for symmetry (e.g., 2x increase is log2(2)=1, 2x decrease is log2(0.5)=-1)
    
    # Avoid division by zero by adding epsilon
    # If a feature is zero in both, it becomes 1/1, log2(1) = 0 (no change).
    # If a feature is zero in control but present in group, it becomes X/epsilon (very high fold change).
    # If a feature is present in control but zero in group, it becomes epsilon/X (very low fold change).
    
    log2_fold_changes = np.log2(
        (relative_abundance_for_group + epsilon) / (relative_abundance_for_control + epsilon)
    )
    
    # Filter for "meaningful" changes based on a log2 fold change threshold (e.g., abs(log2FC) > 1 means 2-fold change)
    meaningful_changes = log2_fold_changes[abs(log2_fold_changes) > 1].sort_values(ascending=False)
    
    if not meaningful_changes.empty:
        print(f"\nGroup {current_group_id_str} vs Control:")
        print(f"  Log2 Fold Changes (Group {current_group_id_str} vs Control):")
        print(f"  Top 10 features with >2x change (log2FC > 1 or < -1):")
        print(meaningful_changes.head(10))
        print(f"  Bottom 10 features with >2x change (log2FC > 1 or < -1):")
        print(meaningful_changes.tail(10))
    else:
        print(f"  No features with >2x change (log2FC > 1 or < -1) found for Group {current_group_id_str} vs Control.")

    # Store these descriptive fold changes if needed for later analysis/plotting
    descriptive_comparison_results[current_group_id_str] = log2_fold_changes


    
    
    print(f"  Aggregated counts for Group {str(current_group_id_str)}. Total features summed: {len(summed_bacteria_for_group)}")

# NEW: Optional: Convert descriptive comparison results to a DataFrame
# final_descriptive_df = pd.DataFrame.from_dict(descriptive_comparison_results)
# final_descriptive_df.index.name = 'Feature'
# for col in final_descriptive_df.columns:
#     for feature in final_descriptive_df.index:
#         if abs(final_descriptive_df.at[feature, col]) > 5:
#             print(f"Feature {feature} in Group {col} has a log2 fold change > 5, which is very high.")
# print("\nFinal Descriptive Log2 Fold Changes (aggregated relative abundances):")
# print(final_descriptive_df.head())
# print(f"Shape of descriptive results: {final_descriptive_df.shape}")

# You can save this DataFrame if you want
# final_descriptive_df.to_csv('aggregated_relative_abundance_log2_fold_changes.csv')

print("\n--- Descriptive Comparison Complete ---")
print("\nRemember: These are descriptive comparisons on aggregated data. For statistical inference, use ANCOM-BC or DESeq2 on original sample-level counts.")