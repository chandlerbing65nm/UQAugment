import numpy as np
import torch
import os
from tqdm import tqdm
import random
from scipy.spatial import KDTree

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def compute_ece(probs, labels, n_bins=20):  # changed from 10 to 20
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

def compute_brier(probs, labels, num_classes=None):
    """Compute Brier Score"""
    if num_classes is None:
        num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return brier

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

def compute_accuracy(probs, labels):
    """
    Compute classification accuracy.
    Args:
        probs (ndarray): shape (N, C), predicted probabilities for each sample.
        labels (ndarray): shape (N,), true class indices for each sample.
    Returns:
        float: The accuracy score.
    """
    predictions = probs.argmax(axis=1)
    return np.mean(predictions == labels)

def subsample_first_50_per_class(probs_list, targets):
    """Return subsampled probs and targets where only first 50 samples per class are kept"""
    num_classes = np.max(targets) + 1
    class_counts = {i: 0 for i in range(num_classes)}
    indices = []

    for idx, label in enumerate(targets):
        if class_counts[label] < 50:
            indices.append(idx)
            class_counts[label] += 1
        if all(v >= 50 for v in class_counts.values()):
            break

    # Apply to all model probs
    subsampled_probs_list = [p[indices] for p in probs_list]
    subsampled_targets = targets[indices]

    return subsampled_probs_list, subsampled_targets


def load_npz_files_for_ensembling(folder_path):
    """Load predictions and targets from .npz files"""
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
    test_targets_list = []
    aug_preds_list = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        test_targets_list.append(data["all_test_targets"])
        aug_preds_list.append(data["all_mc_preds"])

    return test_targets_list, aug_preds_list, file_names

def check_test_target_consistency(test_targets_list, file_names):
    """Verify all test targets are identical"""
    reference_targets = test_targets_list[0]
    for idx, current_targets in enumerate(test_targets_list[1:], start=1):
        if not np.array_equal(reference_targets, current_targets):
            raise ValueError(f"Mismatch in 'all_test_targets' between files '{file_names[0]}' and '{file_names[idx]}'.")
    print("✅ All 'all_test_targets' arrays are identical across all augmentations.")

class PreferenceBasedEnsembleOptimizer:
    def __init__(self, aug_preds_list, test_targets, n_models, dataset_name, model_name, device='cuda'):
        self.aug_preds_list = [torch.tensor(p, device=device) for p in aug_preds_list]
        self.test_targets = torch.tensor(test_targets, device=device)
        self.n_models = n_models
        self.device = device
        self.dataset_name = dataset_name
        self.model_name = model_name
        
        # Initialize action space (small weight adjustments)
        self.action_space = self._create_action_space()
        self.n_actions = len(self.action_space)
        
        # Initialize policy (uniform probabilities)
        self.policy = torch.ones((self.n_actions), device=device) / self.n_actions
        
        # Initialize action preferences
        self.action_preferences = torch.ones((self.n_actions, self.n_actions), device=device) * 0.5
        
        # Store best weights and metrics
        self.best_weights = torch.ones(n_models, device=device) / n_models
        self.best_nll = float('inf')
        
        # For binned state comparison
        self.weight_bins = {}
        self.bin_size = 0.05  # Round weights to 2 decimal places
        
        # For decaying exploration
        self.initial_exploration = 0.5
        self.final_exploration = 0.01
        self.exploration_decay = 0.99

        # Initialize history tracking
        self.history = {
            'weights': [],
            'accuracy': [],
            'nll': [],
            'ece': [],
            'brier': [],
            'exploration_rate': [],
            'augmentation_names': [self._extract_aug_name(f) for f in file_names]
        }

    def _extract_aug_name(self, filename):
        """Extract augmentation name from filename"""
        for aug in ["time_mask", "band_stop_filter", "gaussian_noise", "time_stretch", "pitch_shift"]:
            if aug in filename:
                return aug
        return "base"

    def _create_action_space(self):
        """Create action space as small weight adjustments"""
        actions = []
        step = 0.05  # Adjustment step size
        
        # Create actions that increase one weight and decrease others proportionally
        for i in range(self.n_models):
            action = torch.zeros(self.n_models, device=self.device)
            action[i] = step
            # Normalize to maintain sum=1
            action = action - step/self.n_models
            actions.append(action)
            
            # Also create actions that decrease weight
            action = torch.zeros(self.n_models, device=self.device)
            action[i] = -step
            action = action + step/self.n_models
            actions.append(action)
            
        return actions
    
    def _get_binned_state(self, weights):
        """Round weights to create binned states for comparison"""
        binned = torch.round(weights / self.bin_size) * self.bin_size
        return tuple(binned.cpu().numpy().round(2))
    
    def evaluate_weights(self, weights):
        """Evaluate current weights by computing all metrics"""
        weights = weights / weights.sum()
        ensemble_preds = torch.zeros_like(self.aug_preds_list[0].mean(dim=-1))

        for i in range(self.n_models):
            ensemble_preds += weights[i] * self.aug_preds_list[i].mean(dim=-1)

        ensemble_preds_np = ensemble_preds.cpu().numpy()
        targets_np = self.test_targets.cpu().numpy()

        accuracy = compute_accuracy(ensemble_preds_np, targets_np)
        nll = compute_nll(ensemble_preds_np, targets_np)
        ece = compute_ece(ensemble_preds_np, targets_np)
        brier = compute_brier(ensemble_preds_np, targets_np)

        return accuracy, nll, ece, brier

    def scalarized_objective(self, nll):
        """Use NLL as the optimization objective"""
        return nll  # Lower is better

    
    def generate_trajectory(self, max_steps=10, iteration=0):
        """Generate trajectory with decaying exploration"""
        weights = torch.ones(self.n_models, device=self.device) / self.n_models
        trajectory = []
        
        # Calculate current exploration rate
        exploration_rate = max(
            self.final_exploration,
            self.initial_exploration * (self.exploration_decay ** iteration)
        )
        
        for _ in range(max_steps):
            # Epsilon-greedy exploration with decay
            if random.random() < exploration_rate:
                action_idx = random.randint(0, self.n_actions - 1)
            else:
                action_idx = torch.multinomial(self.policy, 1).item()
            
            adjustment = self.action_space[action_idx]
            new_weights = weights + adjustment
            new_weights = torch.clamp(new_weights, min=0.01)
            new_weights = new_weights / new_weights.sum()
            
            trajectory.append((weights.clone(), action_idx, new_weights.clone()))
            weights = new_weights.clone()
        
        return trajectory
    
    def update_policy(self, trajectories, preferences):
        """Update policy using softmax normalization of action preferences"""
        # Update action preferences based on decisive states
        for (traj1, traj2) in preferences:
            # Get binned states for both trajectories
            binned_states1 = [self._get_binned_state(w) for w, _, _ in traj1]
            binned_states2 = [self._get_binned_state(w) for w, _, _ in traj2]
            
            # Find overlapping binned states
            overlapping_bins = set(binned_states1) & set(binned_states2)
            
            for bin_state in overlapping_bins:
                # Find the first occurrence in each trajectory
                idx1 = next(i for i, (w, _, _) in enumerate(traj1) 
                         if self._get_binned_state(w) == bin_state)
                idx2 = next(i for i, (w, _, _) in enumerate(traj2) 
                         if self._get_binned_state(w) == bin_state)
                
                a1 = traj1[idx1][1]  # Action index in trajectory 1
                a2 = traj2[idx2][1]  # Action index in trajectory 2
                
                # Update preference
                self.action_preferences[a1, a2] += 0.1
                self.action_preferences[a2, a1] = max(0, self.action_preferences[a2, a1] - 0.1)
        
        # Direct softmax normalization (replaces pairwise coupling)
        self.policy = torch.softmax(self.action_preferences.mean(dim=1), dim=0)

    def _save_history(self):
        """Save history to a numpy file"""
        os.makedirs('history', exist_ok=True)
        filename = f"history/{self.dataset_name}_{self.model_name}_history.npz"
        np.savez(
            filename,
            weights=np.array(self.history['weights']),
            accuracy=np.array(self.history['accuracy']),
            nll=np.array(self.history['nll']),
            ece=np.array(self.history['ece']),
            brier=np.array(self.history['brier']),
            exploration_rate=np.array(self.history['exploration_rate']),
            augmentation_names=np.array(self.history['augmentation_names'])
        )
        print(f"Saved optimization history to {filename}")

    def optimize(self, n_iterations=20, n_trajectories=5, max_steps=5):
        # Store initial uniform weights first
        initial_weights = torch.ones(self.n_models, device=self.device) / self.n_models
        initial_accuracy, initial_nll, initial_ece, initial_brier = self.evaluate_weights(initial_weights)
        
        self.history['weights'].append(initial_weights.cpu().numpy())
        self.history['accuracy'].append(initial_accuracy)
        self.history['nll'].append(initial_nll)
        self.history['ece'].append(initial_ece)
        self.history['brier'].append(initial_brier)
        self.history['exploration_rate'].append(self.initial_exploration)

        """Run the optimization process"""
        for iteration in tqdm(range(n_iterations), desc="Optimizing ensemble weights"):
            # Generate trajectories
            trajectories = [self.generate_trajectory(max_steps, iteration) 
                          for _ in range(n_trajectories)]
            
            # Evaluate trajectories and create preferences
            preferences = []
            traj_scores = []
            
            for traj in trajectories:
                final_weights = traj[-1][2]  # Get final weights
                accuracy, nll, ece, brier = self.evaluate_weights(final_weights)
                score = self.scalarized_objective(nll)
                traj_scores.append(score)
                
                # Update best weights if improved
                if nll < self.best_nll:
                    self.best_nll = nll
                    self.best_weights = final_weights.clone()
            
            # Generate preferences based on NLL scores
            for i in range(len(trajectories)):
                for j in range(i+1, len(trajectories)):
                    if traj_scores[i] < traj_scores[j]:
                        preferences.append((trajectories[i], trajectories[j]))
                    elif traj_scores[j] < traj_scores[i]:
                        preferences.append((trajectories[j], trajectories[i]))
            
            # Update policy if we have preferences
            if preferences:
                self.update_policy(trajectories, preferences)

            # Store history at each iteration
            self.history['weights'].append(self.best_weights.cpu().numpy().copy())
            accuracy, nll, ece, brier = self.evaluate_weights(self.best_weights)
            self.history['accuracy'].append(accuracy)
            self.history['nll'].append(nll)
            self.history['ece'].append(ece)
            self.history['brier'].append(brier)
            self.history['exploration_rate'].append(
                max(self.final_exploration,
                    self.initial_exploration * (self.exploration_decay ** iteration)))

        # Save history to file
        self._save_history()     
        return self.best_weights.cpu().numpy()

def compute_prediction_variance(probs_list, weights):
    """
    Compute the variance of predictions across ensemble models.
    Args:
        probs_list (list): List of prediction arrays from each model
        weights (ndarray): Weights for each model in the ensemble
    Returns:
        float: Average variance across all samples
    """
    # Stack all predictions
    stacked_probs = np.stack([p.mean(axis=-1) for p in probs_list], axis=0)
    
    # Compute weighted mean
    weighted_mean = np.sum(weights[:, np.newaxis, np.newaxis] * stacked_probs, axis=0)
    
    # Compute variance
    variance = np.sum(weights[:, np.newaxis, np.newaxis] * (stacked_probs - weighted_mean[np.newaxis, :, :])**2, axis=0)
    
    # Return mean variance across all samples
    return np.mean(variance)

def compute_prediction_entropy(probs_list, weights):
    """
    Compute the entropy of the ensemble predictions.
    Args:
        probs_list (list): List of prediction arrays from each model
        weights (ndarray): Weights for each model in the ensemble
    Returns:
        float: Average entropy across all samples
    """
    # Stack all predictions
    stacked_probs = np.stack([p.mean(axis=-1) for p in probs_list], axis=0)
    
    # Compute weighted ensemble predictions
    ensemble_probs = np.sum(weights[:, np.newaxis, np.newaxis] * stacked_probs, axis=0)
    
    # Compute entropy for each sample
    entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
    
    # Return mean entropy across all samples
    return np.mean(entropy)

# models:   panns_cnn6, panns_mobilenetv2, ast
# datasets: affia3k, mrsffia, uffia

if __name__ == "__main__":
    dataset = 'uffia'
    model = 'ast'
    folder_path = f"/users/doloriel/work/Repo/UQFishAugment/probs_epistemic/{dataset}/{model}"

    # Load data
    test_targets_list, aug_preds_list, file_names = load_npz_files_for_ensembling(folder_path)

    # Check test target consistency
    check_test_target_consistency(test_targets_list, file_names)

    # Define targets
    all_test_targets = test_targets_list[0]

    # 🔁 Subsample after test_targets are assigned
    aug_preds_list, all_test_targets = subsample_first_50_per_class(aug_preds_list, all_test_targets)

    # Initialize optimizer
    optimizer = PreferenceBasedEnsembleOptimizer(
        aug_preds_list, all_test_targets, len(file_names), 
        dataset_name=f"{dataset}",
        model_name=f"{model}",
        device=device
    )
    # Run optimization
    best_weights = optimizer.optimize(n_iterations=300, n_trajectories=10, max_steps=10)

    # Evaluate best weights
    ensemble_preds = np.zeros_like(aug_preds_list[0].mean(axis=-1))
    for i in range(len(file_names)):
        ensemble_preds += best_weights[i] * aug_preds_list[i].mean(axis=-1)

    # Compute metrics
    accuracy = compute_accuracy(ensemble_preds, all_test_targets)
    nll = compute_nll(ensemble_preds, all_test_targets)
    ece = compute_ece(ensemble_preds, all_test_targets)
    brier = compute_brier(ensemble_preds, all_test_targets)
    pred_variance = compute_prediction_variance(aug_preds_list, best_weights)
    pred_entropy = compute_prediction_entropy(aug_preds_list, best_weights)

    print("\nOptimization Results:")
    print("Best Weights:")
    for name, weight in zip(file_names, best_weights):
        # Extract augmentation type from filename
        aug_type = "unknown"
        for aug in ["time_mask", "band_stop_filter", "gaussian_noise", "time_stretch", "pitch_shift"]:
            if aug in name:
                aug_type = aug
                break
        print(f"- {aug_type}: {weight:.4f}")
    
    print(f"\nMetrics with optimized weights:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"NLL: {nll:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Prediction Variance: {pred_variance:.4f}")
    print(f"Prediction Entropy: {pred_entropy:.4f}")
    print("-" * 50)

    # Compare with uniform weights
    uniform_weights = np.ones(len(file_names)) / len(file_names)
    uniform_preds = np.zeros_like(aug_preds_list[0].mean(axis=-1))
    for i in range(len(file_names)):
        uniform_preds += uniform_weights[i] * aug_preds_list[i].mean(axis=-1)

    uniform_accuracy = compute_accuracy(uniform_preds, all_test_targets)
    uniform_nll = compute_nll(uniform_preds, all_test_targets)
    uniform_ece = compute_ece(uniform_preds, all_test_targets)
    uniform_brier = compute_brier(uniform_preds, all_test_targets)
    uniform_pred_variance = compute_prediction_variance(aug_preds_list, uniform_weights)
    uniform_pred_entropy = compute_prediction_entropy(aug_preds_list, uniform_weights)

    print("\nUniform Weights Comparison:")
    print("Uniform Weights:")
    for name, weight in zip(file_names, uniform_weights):
        # Extract augmentation type from filename
        aug_type = "unknown"
        for aug in ["time_mask", "band_stop_filter", "gaussian_noise", "time_stretch", "pitch_shift"]:
            if aug in name:
                aug_type = aug
                break
        print(f"- {aug_type}: {weight:.4f}")
    
    print(f"\nMetrics with uniform weights:")
    print(f"Accuracy: {uniform_accuracy:.4f}")
    print(f"NLL: {uniform_nll:.4f}")
    print(f"ECE: {uniform_ece:.4f}")
    print(f"Brier Score: {uniform_brier:.4f}")
    print(f"Prediction Variance: {uniform_pred_variance:.4f}")
    print(f"Prediction Entropy: {uniform_pred_entropy:.4f}")
    print("-" * 50)