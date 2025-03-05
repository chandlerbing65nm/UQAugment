import numpy as np
import os
from tqdm import tqdm

def compute_ece(probs, labels, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).
    Args:
        probs (ndarray): shape (N, C), predicted probabilities for each sample.
        labels (ndarray): shape (N,), true class indices for each sample.
        n_bins (int): Number of bins to use for ECE.
    Returns:
        float: The ECE value.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        in_bin = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        if len(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_prob = len(in_bin) / len(probs)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_prob

    return ece

def compute_nll(probs, labels):
    """
    Compute the Negative Log-Likelihood (NLL).
    Args:
        probs (ndarray): shape (N, C), predicted probabilities for each sample.
        labels (ndarray): shape (N,), true class indices for each sample.
    Returns:
        float: The average NLL across all samples.
    """
    eps = 1e-12  # for numerical stability
    true_class_probs = probs[np.arange(len(labels)), labels]
    nll = -np.mean(np.log(true_class_probs + eps))
    return nll

def load_and_concatenate_npz_files(folder_path):
    """
    Load and concatenate all_test_targets and all_mc_preds from all .npz files in the folder.
    Args:
        folder_path (str): Path to the folder containing .npz files.
    Returns:
        all_test_targets (ndarray): Concatenated true labels.
        all_mc_preds (ndarray): Concatenated predicted probabilities.
        file_names (list): List of file names for reference.
        file_indices (list): List of indices indicating which file each sample belongs to.
    """
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
    all_test_targets = []
    all_mc_preds = []
    file_indices = []

    for idx, file_name in enumerate(file_names):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        all_test_targets.append(data["all_test_targets"])
        all_mc_preds.append(data["all_mc_preds"])
        file_indices.extend([idx] * len(data["all_test_targets"]))

    # Concatenate along the first axis (samples)
    all_test_targets = np.concatenate(all_test_targets, axis=0)
    all_mc_preds = np.concatenate(all_mc_preds, axis=0)

    return all_test_targets, all_mc_preds, file_names, file_indices

def compute_metrics_for_combinations(all_test_targets, all_mc_preds, file_names, file_indices):
    """
    Compute ECE and NLL for the full concatenated data and for all combinations where one .npz file is removed.
    Args:
        all_test_targets (ndarray): Concatenated true labels.
        all_mc_preds (ndarray): Concatenated predicted probabilities.
        file_names (list): List of file names for reference.
        file_indices (list): List of indices indicating which file each sample belongs to.
    """
    # List of valid augmentation strings to search for
    valid_augmentations = ["specaugment", "specmix", "gaussian_noise", "time_stretch", "pitch_shift"]

    # Compute metrics for the full concatenated data
    mean_preds = all_mc_preds.mean(axis=-1)  # Average over MC runs
    ece_full = compute_ece(mean_preds, all_test_targets)
    nll_full = compute_nll(mean_preds, all_test_targets)
    print(f"All augmentation: ECE = {ece_full:.4f}, NLL = {nll_full:.4f}")
    print("-" * 50)

    # Compute metrics for all combinations where one .npz file is removed
    for i, file_name in enumerate(file_names):
        # Extract the augmentation name from the filename
        found_augmentations = [aug for aug in valid_augmentations if aug in file_name]

        # Check if more than one augmentation is found
        if len(found_augmentations) > 1:
            raise ValueError(f"Filename '{file_name}' contains multiple augmentations: {found_augmentations}. Only one is allowed.")
        elif len(found_augmentations) == 0:
            raise ValueError(f"Filename '{file_name}' does not contain any valid augmentation. Expected one of: {valid_augmentations}.")
        else:
            augmentation_name = found_augmentations[0]

        # Create a mask to exclude samples from the i-th file
        mask = np.array(file_indices) != i

        # Apply the mask to exclude the i-th file's data
        test_targets_subset = all_test_targets[mask]
        mc_preds_subset = all_mc_preds[mask]

        # Compute metrics for the subset
        mean_preds_subset = mc_preds_subset.mean(axis=-1)
        ece_subset = compute_ece(mean_preds_subset, test_targets_subset)
        nll_subset = compute_nll(mean_preds_subset, test_targets_subset)

        print(f"Removed augmentation: {augmentation_name}")
        print(f"Subset: ECE = {ece_subset:.4f}, NLL = {nll_subset:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    # User-defined folder path
    folder_path = "/users/doloriel/work/Repo/FrameMixer/probs/mrsffia/panns_mobilenetv2"

    # Load and concatenate data from all .npz files
    all_test_targets, all_mc_preds, file_names, file_indices = load_and_concatenate_npz_files(folder_path)

    # Compute metrics for the full dataset and all combinations
    compute_metrics_for_combinations(all_test_targets, all_mc_preds, file_names, file_indices)