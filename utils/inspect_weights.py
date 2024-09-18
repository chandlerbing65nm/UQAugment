import torch
from pprint import pprint

def save_pretty_weights(model_path, output_path=None):
    # Load the model weights from the .pth file
    model_weights = torch.load(model_path)

    # Create a dictionary for pretty-printing that handles different data types
    pretty_weights = {}
    for k, v in model_weights.items():
        if isinstance(v, torch.Tensor):  # If it's a tensor, show the shape
            pretty_weights[k] = f"Tensor of shape {v.shape}"
        elif isinstance(v, dict):  # If it's a dictionary (e.g., model or optimizer state)
            if k == 'model':
                # Inspect the 'model' key more deeply
                print("\n'{}' contains the following layers:".format(k))
                for layer_name, layer_weights in v.items():
                    if isinstance(layer_weights, torch.Tensor):
                        print(f"Layer '{layer_name}': Tensor of shape {layer_weights.shape}")
                    else:
                        print(f"Layer '{layer_name}': {type(layer_weights)}")
            pretty_weights[k] = f"Dictionary with {len(v)} keys"
        else:  # For other types like int, float, etc.
            pretty_weights[k] = str(v)

    # Pretty print the model weights (keys and their summaries)
    pprint(pretty_weights)

    # if output_path:
    #     # Optionally, save the pretty-printed weights into a text file
    #     with open(output_path, 'w') as f:
    #         pprint(pretty_weights, stream=f)
    #     print(f"Pretty-printed weights saved to {output_path}")

        
# Example usage
model_path = './weights/Cnn6_mAP=0.343.pth'  # Path to the model weights file
output_path = 'pretty_weights.txt'  # Path to save the pretty-printed weights (optional)

save_pretty_weights(model_path, output_path)
