import pandas as pd

def convert_to_relative_abundance(input_csv_path, output_csv_path, sample_id_column=None):
    """
    Converts a CSV of absolute microbial counts to relative abundances per sample.

    Args:
        input_csv_path (str): Path to the input CSV file with absolute counts.
                              Assumes samples are rows and taxa/features are columns.
        output_csv_path (str): Path where the output CSV file with relative abundances will be saved.
        sample_id_column (str or int, optional): The name of the column containing sample IDs,
                                                  or its 0-based index. If None, the first column
                                                  is assumed to be the index.
    """
    try:
        # Load the data
        if sample_id_column is not None:
            # If a specific ID column is provided, read it as part of the data
            df_counts = pd.read_csv(input_csv_path)
            # Set it as index and drop the column
            df_counts = df_counts.set_index(sample_id_column, drop=True)
        else:
            # Assume the first column is the index (SampleID) if no specific column is given
            df_counts = pd.read_csv(input_csv_path, index_col=0)

        print(f"Loaded data from: {input_csv_path}")
        print("Original data head:")
        print(df_counts.head())
        print("\nOriginal data shape:", df_counts.shape)

        # Calculate the sum of counts for each sample (row-wise sum)
        # We sum across axis=1 (columns) for each row
        sample_total_reads = df_counts.sum(axis=1)

        print("\nTotal reads per sample (first 5):")
        print(sample_total_reads.head())

        # Handle potential samples with zero total reads to avoid division by zero
        zero_total_reads_samples = sample_total_reads[sample_total_reads == 0].index
        if not zero_total_reads_samples.empty:
            print(f"\nWarning: The following samples have 0 total reads and will result in NaNs/zeros:")
            for sample in zero_total_reads_samples:
                print(f"- {sample}")
            # For these samples, relative abundances will correctly be 0/NaN, but it's good to warn.
            # If you want to remove them, add: df_counts = df_counts[sample_total_reads != 0]

        # Divide each count by its respective sample's total reads to get relative abundance
        # `axis=0` ensures that division happens row-wise (each row's values divided by that row's sum)
        df_relative_abundance = df_counts.div(sample_total_reads, axis=0)

        # Verify that each row now sums to approximately 1
        print("\nVerifying row sums (first 5, should be ~1.0):")
        print(df_relative_abundance.sum(axis=1).head())

        # Save the transformed data to a new CSV
        df_relative_abundance.to_csv(output_csv_path)
        print(f"\nRelative abundance data saved to: {output_csv_path}")
        print("Transformed data head:")
        print(df_relative_abundance.head())

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Configuration ---
# Set the path to your input CSV file
input_file = 'microbiome.csv' 
# Set the path for the output relative abundance CSV file
output_file = 'microbiome_relative_abundance.csv'

# Optional: If your Sample ID column has a name other than the first column
# For example, if it's named 'SampleID', use sample_id_col='SampleID'
# If the first column is indeed the SampleID, leave as None or use index 0
sample_id_col = None # Or 'SampleID', or 0 if it's the first column (default)

# --- Run the conversion ---
if __name__ == "__main__":
    convert_to_relative_abundance(input_file, output_file, sample_id_column=sample_id_col)