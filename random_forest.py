import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt

# --- 0. Configuration & Helper Function ---
RANDOM_STATE = 42 # For reproducibility
TEST_SIZE = 0.2    # Proportion of data to use for testing

def plot_micro_averaged_pr_curve(y_true_multiclass, y_scores_proba, n_unique_classes, model_label='Model', ax=None):
    """
    Plots the micro-averaged Precision-Recall curve for multi-class classification.
    y_true_multiclass: 1D array of true class labels (0 to k-1).
    y_scores_proba: 2D array of probability scores (n_samples, n_classes).
    n_unique_classes: The total number of unique classes.
    model_label: Label for the curve in the legend.
    ax: Matplotlib axis to plot on.
    """
    # Binarize the output against all known classes
    # Ensure classes for binarization are 0, 1, ..., n_unique_classes-1
    y_true_binarized = label_binarize(y_true_multiclass, classes=np.arange(n_unique_classes))

    # If y_true_binarized has fewer columns than n_unique_classes, it means some classes
    # were not present in this specific y_true_multiclass split.
    # Pad with zeros if necessary for consistent .ravel() with y_scores_proba.
    if y_true_binarized.shape[1] < n_unique_classes:
        padding = np.zeros((y_true_binarized.shape[0], n_unique_classes - y_true_binarized.shape[1]))
        y_true_binarized = np.hstack((y_true_binarized, padding))


    # Check if any positive instances exist after binarization (globally)
    if np.sum(y_true_binarized) == 0:
        print(f"Warning: No positive instances found in y_true_binarized for model '{model_label}'. Micro-AUPR will be undefined or 0.")
        aupr_micro = 0.0
        # Plot a dummy point or nothing
        current_ax = ax if ax else plt.gca()
        current_ax.plot([0, 1], [0.5, 0.5], linestyle='--', label=f'{model_label} (Micro AUPR = {aupr_micro:.3f} - No positive samples)')
        return aupr_micro

    precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), y_scores_proba.ravel())
    aupr_micro = auc(recall, precision)

    current_ax = ax if ax else plt.gca()
    current_ax.plot(recall, precision, label=f'{model_label} (Micro AUPR = {aupr_micro:.3f})')
    
    # Common plot settings (if not using a shared ax, apply them here)
    if not ax:
        plt.xlabel('Recall (Micro-averaged)')
        plt.ylabel('Precision (Micro-averaged)')
        plt.title('Micro-averaged Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
    return aupr_micro


# --- Configuration for Data Loading ---
METADATA_FILE = 'metadata.csv'
MICROBIOME_FILE = 'microbiome_relative_abundance.csv'
METABOLOMICS_FILE = 'serum_lipo.csv'

SAMPLE_ID_COL = 'SampleID'
DISEASE_STATUS_COL = 'PATGROUPFINAL_C' # This column should now contain numbers 1-7

# --- 1. Load and Prepare Data ---
print("--- 1. Loading and Preparing Data ---")
le = LabelEncoder() # Initialize LabelEncoder

try:
    # Load metadata
    metadata_df = pd.read_csv(METADATA_FILE)
    if SAMPLE_ID_COL not in metadata_df.columns:
        raise ValueError(f"Sample ID column '{SAMPLE_ID_COL}' not found in {METADATA_FILE}.")
    if DISEASE_STATUS_COL not in metadata_df.columns:
        raise ValueError(f"Disease status column '{DISEASE_STATUS_COL}' not found in {METADATA_FILE}.")

    metadata_df = metadata_df.set_index(SAMPLE_ID_COL)
    
    # Extract and process target variable y for multi-class
    y_raw = metadata_df[DISEASE_STATUS_COL].copy()
    y_raw = y_raw.dropna() # Drop samples with missing disease status
    
    if y_raw.empty:
        raise ValueError(f"Target column '{DISEASE_STATUS_COL}' is empty or all NaNs.")

    # Ensure the disease status column is numeric if it's not already
    if not pd.api.types.is_numeric_dtype(y_raw):
        print(f"Warning: Disease status column '{DISEASE_STATUS_COL}' is not numeric. Attempting conversion...")
        y_raw = pd.to_numeric(y_raw, errors='coerce').dropna()
        if y_raw.empty:
            raise ValueError(f"Could not convert '{DISEASE_STATUS_COL}' to numeric or all values became NaN.")
    
    # Use LabelEncoder to encode the disease status
    y_encoded = le.fit_transform(y_raw)
    y = pd.Series(y_encoded, index=y_raw.index, name='disease_status_encoded')
    
    N_CLASSES = len(le.classes_)
    print(f"Identified {N_CLASSES} unique classes in '{DISEASE_STATUS_COL}'.")
    print(f"Original class labels: {le.classes_}")
    print(f"Encoded class labels (0 to {N_CLASSES-1}): {np.unique(y_encoded)}")
    if N_CLASSES < 2:
        raise ValueError("Found less than 2 classes for modeling. Multi-class classification requires at least 2 classes.")

    print(f"Target 'y' processed. {len(y)} samples. Encoded class distribution:\n{y.value_counts(normalize=True).sort_index()}")

    # Load microbiome data
    microbiome_df_raw = pd.read_csv(MICROBIOME_FILE)
    if SAMPLE_ID_COL not in microbiome_df_raw.columns:
        microbiome_df_raw = microbiome_df_raw.rename(columns={microbiome_df_raw.columns[0]: SAMPLE_ID_COL})
        if SAMPLE_ID_COL not in microbiome_df_raw.columns: # check again after potential rename
             raise ValueError(f"Sample ID column '{SAMPLE_ID_COL}' not found in {MICROBIOME_FILE}.")
    X_microbiome = microbiome_df_raw.set_index(SAMPLE_ID_COL)
    X_microbiome = X_microbiome.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    print(f"Microbiome data X_microbiome loaded with shape: {X_microbiome.shape}")

    # Load metabolomic data
    metabolome_df_raw = pd.read_csv(METABOLOMICS_FILE)
    if SAMPLE_ID_COL not in metabolome_df_raw.columns:
        metabolome_df_raw = metabolome_df_raw.rename(columns={metabolome_df_raw.columns[0]: SAMPLE_ID_COL})
        if SAMPLE_ID_COL not in metabolome_df_raw.columns: # check again
            raise ValueError(f"Sample ID column '{SAMPLE_ID_COL}' not found in {METABOLOMICS_FILE}.")
    X_metabolome = metabolome_df_raw.set_index(SAMPLE_ID_COL)
    X_metabolome = X_metabolome.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    print(f"Metabolomic data X_metabolome loaded with shape: {X_metabolome.shape}")

    # Align data
    common_samples = y.index.intersection(X_microbiome.index).intersection(X_metabolome.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples found across all datasets.")
    
    y = y.loc[common_samples].sort_index()
    X_microbiome = X_microbiome.loc[common_samples].sort_index()
    X_metabolome = X_metabolome.loc[common_samples].sort_index()

    print(f"\nData aligned. Number of common samples: {len(common_samples)}")
    print(f"Final shape of y: {y.shape}")
    print(f"Final shape of X_microbiome: {X_microbiome.shape}")
    print(f"Final shape of X_metabolome: {X_metabolome.shape}")

    # Basic missing value handling for features
    X_microbiome = X_microbiome.fillna(0)
    X_metabolome = X_metabolome.fillna(0)
    X_microbiome = X_microbiome.dropna(axis=1, how='all')
    X_metabolome = X_metabolome.dropna(axis=1, how='all')

    if X_microbiome.empty or X_metabolome.empty:
        raise ValueError("One of the feature matrices is empty after NaN handling and alignment.")
    
    N_CLASSES_FINAL = y.nunique() # Re-check N_CLASSES after alignment
    if N_CLASSES_FINAL != N_CLASSES:
        print(f"Warning: Number of unique classes changed after sample alignment from {N_CLASSES} to {N_CLASSES_FINAL}.")
        N_CLASSES = N_CLASSES_FINAL
        # Re-fit LabelEncoder on the aligned 'y' to ensure classes map correctly from 0..N_CLASSES-1
        y_raw_aligned = metadata_df.loc[y.index, DISEASE_STATUS_COL].copy() # Get original labels for aligned samples
        y_encoded_aligned = le.fit_transform(y_raw_aligned) # Re-fit and transform
        y = pd.Series(y_encoded_aligned, index=y.index, name='disease_status_encoded')
        print(f"Re-encoded class labels (0 to {N_CLASSES-1}): {np.unique(y_encoded_aligned)}")
        print(f"New original class labels after alignment: {le.classes_}")


    if N_CLASSES < 2:
        raise ValueError("Target variable 'y' has less than 2 classes after all processing.")
    min_samples_per_class_for_stratify = max(2, int(0.1 * N_CLASSES)) # heuristic for stratification
    if y.value_counts().min() < min_samples_per_class_for_stratify:
        print(f"Warning: Very few samples in one or more classes ({y.value_counts().min()}). Stratification might be problematic.")
    print(f"Final encoded class distribution for modeling:\n{y.value_counts(normalize=True).sort_index()}")

except FileNotFoundError as e:
    print(f"Error: File not found. {e}"); exit()
except ValueError as e:
    print(f"Error during data loading/preparation: {e}"); exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}"); exit()

# --- Shared plotting setup ---
fig, ax = plt.subplots(figsize=(10, 8))


# --- 2. Model 1: Microbiome Data Only ---
print("\n--- Model 1: Microbiome Data Only ---")
X_train_mic, X_test_mic, y_train, y_test = train_test_split(
    X_microbiome, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"Microbiome model: Training on {X_train_mic.shape[0]} samples, testing on {X_test_mic.shape[0]} samples.")
rf_mic = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, class_weight='balanced')
rf_mic.fit(X_train_mic, y_train)
y_scores_proba_mic = rf_mic.predict_proba(X_test_mic)

aupr_macro_mic = average_precision_score(y_test, y_scores_proba_mic, average='macro')
aupr_weighted_mic = average_precision_score(y_test, y_scores_proba_mic, average='weighted')
print(f"Macro AUPR (Microbiome Only): {aupr_macro_mic:.4f}")
print(f"Weighted AUPR (Microbiome Only): {aupr_weighted_mic:.4f}")
plot_micro_averaged_pr_curve(y_test, y_scores_proba_mic, N_CLASSES, model_label='Microbiome Only', ax=ax)


# --- 3. Model 2: Microbiome + Metabolomic Data ---
print("\n--- Model 2: Microbiome + Metabolomic Data ---")
X_combined = pd.concat([X_microbiome, X_metabolome], axis=1)
print(f"Combined data X_combined shape: {X_combined.shape}")
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    X_combined, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
# Sanity check: y_test should be identical to y_test_comb
if not y_test.equals(y_test_comb):
     print("Warning: y_test and y_test_comb are not identical. Check splitting logic.")

print(f"Combined model: Training on {X_train_comb.shape[0]} samples, testing on {X_test_comb.shape[0]} samples.")
rf_comb = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, class_weight='balanced')
rf_comb.fit(X_train_comb, y_train_comb)
y_scores_proba_comb = rf_comb.predict_proba(X_test_comb)

aupr_macro_comb = average_precision_score(y_test_comb, y_scores_proba_comb, average='macro')
aupr_weighted_comb = average_precision_score(y_test_comb, y_scores_proba_comb, average='weighted')
print(f"Macro AUPR (Microbiome + Metabolome): {aupr_macro_comb:.4f}")
print(f"Weighted AUPR (Microbiome + Metabolome): {aupr_weighted_comb:.4f}")
plot_micro_averaged_pr_curve(y_test_comb, y_scores_proba_comb, N_CLASSES, model_label='Microbiome + Metabolome', ax=ax)

# Finalize plot
ax.set_xlabel('Recall (Micro-averaged)')
ax.set_ylabel('Precision (Micro-averaged)')
ax.set_title('Micro-averaged Precision-Recall Curves')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# --- 4. Assess Improvement (using Macro AUPR for primary comparison) ---
print("\n--- Assessment of Improvement (based on Macro AUPR) ---")
print(f"Macro AUPR (Microbiome Only): {aupr_macro_mic:.4f}")
print(f"Macro AUPR (Microbiome + Metabolome): {aupr_macro_comb:.4f}")

improvement_macro = aupr_macro_comb - aupr_macro_mic
if improvement_macro > 0.0001:
    print(f"Incorporating metabolomic data improved Macro AUPR by {improvement_macro:.4f}.")
elif improvement_macro < -0.0001:
    print(f"Incorporating metabolomic data decreased Macro AUPR by {abs(improvement_macro):.4f}.")
else:
    print("Incorporating metabolomic data did not significantly change the Macro AUPR.")

print("\n--- Assessment of Improvement (based on Weighted AUPR) ---")
print(f"Weighted AUPR (Microbiome Only): {aupr_weighted_mic:.4f}")
print(f"Weighted AUPR (Microbiome + Metabolome): {aupr_weighted_comb:.4f}")
improvement_weighted = aupr_weighted_comb - aupr_weighted_mic
if improvement_weighted > 0.0001:
    print(f"Incorporating metabolomic data improved Weighted AUPR by {improvement_weighted:.4f}.")

print("\nNote: This is a 'naive' model. For more robust conclusions, consider:")
print("- Cross-validation (e.g., StratifiedKFold).")
print("- Hyperparameter tuning.")
print("- Feature selection/engineering.")
print("- Advanced imputation if fillna(0) is too simplistic.")
print("- Statistical tests for comparing AUPR values.")
print("- Examining per-class AUPR scores if specific class performance is critical.")
