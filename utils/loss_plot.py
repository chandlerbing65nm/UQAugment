import matplotlib.pyplot as plt

# Function to extract values from the text without using regex
def extract_metrics(file_path):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove any surrounding whitespace


            # Process lines with 'Epoch [' for training metrics
            if 'Epoch [' in line and 'Total Loss:' in line and 'Accuracy:' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        total_loss = float(parts[1].split(':')[1].strip())
                        accuracy = float(parts[2].split(':')[1].strip())
                        train_losses.append(total_loss)
                        train_accuracies.append(accuracy)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping line due to error in extracting training metrics: {line}")
                        continue

            # Process lines with 'Val Loss:' for validation metrics
            elif 'Val Loss:' in line and 'Val Accuracy:' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        val_loss = float(parts[1].split(':')[1].strip())
                        val_accuracy = float(parts[2].split(':')[1].strip())
                        val_losses.append(val_loss)
                        val_accuracies.append(val_accuracy)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping line due to error in extracting validation metrics: {line}")
                        continue

    return train_losses, val_losses, train_accuracies, val_accuracies


# Function to plot and save the figure
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_path):
    epochs = range(1, len(train_losses) + 1)

    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', marker='o')
    ax1.set_title('Train Loss vs Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(epochs, val_accuracies, label='Val Accuracy', marker='o')
    ax2.set_title('Train Accuracy vs Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Main function to run the process
def main():
    # Path to the input txt file
    file_path = '/mnt/users/chadolor/work/slurm-14810178.out'  # Update with your file path
    output_path = 'training_validation_metrics.png'

    # Extract metrics from the file
    train_losses, val_losses, train_accuracies, val_accuracies = extract_metrics(file_path)

    # import ipdb; ipdb.set_trace() 
    # print(train_losses[:3])
    # print(val_losses[:3])
    # print(train_accuracies[:3])

    # Check if we have enough data to plot
    if len(train_losses) > 0 and len(val_losses) > 0:
        # Plot and save the figure
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_path)
    else:
        print("Insufficient data for plotting")

# Run the main function
if __name__ == '__main__':
    main()