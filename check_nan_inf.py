import numpy as np
import os

def check_file_for_nan_inf(filepath, label):
    """Loads data from a file and checks for NaN/Inf values."""
    print(f"Checking {label} ({filepath})...")
    try:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False # Indicate file was not found

        data = np.loadtxt(filepath, dtype=np.float32)

        if data.size == 0:
            print(f"File is empty: {filepath}")
            return True # File exists and is empty, no NaNs/Infs to find

        num_nans = np.sum(np.isnan(data))
        num_infs = np.sum(np.isinf(data))

        if num_nans == 0 and num_infs == 0:
            print(f"-> OK (No NaNs or Infs)")
        else:
            print(f"-> WARNING: Found {num_nans} NaN(s) and {num_infs} Inf(s).")
        return True # File checked

    except Exception as e:
        print(f"-> ERROR: An error occurred while checking {filepath}: {e}")
        return False # Indicate error during check

def main():
    print("--- Checking C++ Output Files for NaN/Inf ---")
    cpp_files_to_check = [
        ("cpp_mdct_coeffs_0.txt", "C++ MDCT Coeffs 0"),
        ("cpp_mdct_coeffs_1.txt", "C++ MDCT Coeffs 1"),
        ("cpp_mdct_coeffs_2.txt", "C++ MDCT Coeffs 2"),
        ("cpp_imdct_output_0.txt", "C++ IMDCT Output 0"),
        ("cpp_imdct_output_1.txt", "C++ IMDCT Output 1"),
        ("cpp_imdct_output_2.txt", "C++ IMDCT Output 2"),
        ("cpp_reconstructed_output.txt", "C++ Reconstructed Output")
    ]
    for filepath, label in cpp_files_to_check:
        check_file_for_nan_inf(filepath, label)

    print("\n--- Checking Python Output Files for NaN/Inf ---")
    py_files_to_check = [
        ("py_mdct_coeffs_0.txt", "Python MDCT Coeffs 0"),
        ("py_mdct_coeffs_1.txt", "Python MDCT Coeffs 1"),
        ("py_mdct_coeffs_2.txt", "Python MDCT Coeffs 2"),
        ("py_imdct_output_0.txt", "Python IMDCT Output 0"),
        ("py_imdct_output_1.txt", "Python IMDCT Output 1"),
        ("py_imdct_output_2.txt", "Python IMDCT Output 2"),
        ("py_reconstructed_output.txt", "Python Reconstructed Output")
    ]
    for filepath, label in py_files_to_check:
        check_file_for_nan_inf(filepath, label)

    print("\nNaN/Inf check finished.")

if __name__ == "__main__":
    main()
