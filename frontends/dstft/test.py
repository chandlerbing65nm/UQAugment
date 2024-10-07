import torch
import torch.nn as nn
import torch.optim as optim
from frontends.dstft.frontend import ADSTFT, FDSTFT, DSTFT
import matplotlib.pyplot as plt
import torch.nn.functional as F

def main():
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define spectrogram parameters
    sampling_rate = 128000  # 128 kHz
    window_length = 2048
    hop_length = 1024
    support = window_length  # Assuming support size is equal to window length
    pow = 2.0
    win_pow = 2.0

    # 3. Create dummy input
    batch_size = 200  # Example batch size
    signal_length = sampling_rate*2  # Example signal length (e.g., 2 seconds at 128kHz)

    # Create a random tensor to simulate audio signals
    x = torch.randn(batch_size, signal_length).to(device)
    print(f"Input shape: {x.shape}")

    # dstft = ADSTFT(
    #     x=x,
    #     win_length=window_length,
    #     support=support,
    #     stride=hop_length,
    #     pow=pow,
    #     win_pow=win_pow,
    #     win_requires_grad=True,
    #     stride_requires_grad=True,
    #     pow_requires_grad=False,    # Ensure gradient is required
    #     win_p="tf",
    #     win_min=window_length//2,               # Custom minimum window length
    #     win_max=window_length,              # Custom maximum window length
    #     stride_min=hop_length//2,            # Custom minimum stride
    #     stride_max=hop_length,           # Custom maximum stride
    #     sr=sampling_rate,
    # ).to(device)

    dstft = DSTFT(
        x=x,
        win_length=window_length,
        support=support,
        stride=hop_length,
        pow=pow,
        win_pow=win_pow,
        win_requires_grad=True,
        stride_requires_grad=True,
        pow_requires_grad=False,    # Ensure gradient is required
        win_p="t",
        win_min=window_length//2,               # Custom minimum window length
        win_max=window_length,              # Custom maximum window length
        stride_min=hop_length//2,            # Custom minimum stride
        stride_max=hop_length,           # Custom maximum stride
        sr=sampling_rate,
    ).to(device)

    print("DSTFT module initialized.")

    # 5. Define a simple classifier
    # We'll perform global average pooling over frequency and time, then a linear layer
    class Classifier(nn.Module):
        def __init__(self, num_classes=4):
            super(Classifier, self).__init__()
            self.fc = nn.Linear(1025, num_classes)  # Assuming freq_bins=1025

        def forward(self, spec):
            # spec shape: [batch_size, freq_bins, time_frames]
            # Perform global average pooling over time frames
            pooled = spec.mean(dim=2)  # shape: [batch_size, freq_bins]
            logits = self.fc(pooled)   # shape: [batch_size, num_classes]
            return logits

    classifier = Classifier(num_classes=4).to(device)
    print("Classifier module initialized.")

    # 6. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(dstft.parameters()) + list(classifier.parameters()), lr=1e-3
    )

    # 7. Create dummy labels for 4 classes
    num_classes = 4
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    print(f"Labels: {labels}")

    # 8. Training loop for 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # Set modules to training mode
        dstft.train()
        classifier.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through ADSTFT
        spec, stft = dstft(x)
        print(f"Epoch {epoch+1}: Spectrogram shape: {spec.shape}")
        # print(f"Epoch {epoch+1}: STFT shape: {stft.shape}")

        # Forward pass through classifier
        logits = classifier(spec)  # shape: [batch_size, num_classes]

        # Compute loss
        loss = criterion(logits, labels)
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Verify gradients
        # You can uncomment the following lines to see gradient values each epoch
        # Note: Printing gradients for every epoch can clutter the output
        """
        if adstft.win_length.grad is not None:
            print(f"Epoch {epoch+1}: Gradient for window length: {adstft.win_length.grad}")
        else:
            print(f"Epoch {epoch+1}: No gradient for window length.")

        if adstft.strides.grad is not None:
            print(f"Epoch {epoch+1}: Gradient for strides: {adstft.strides.grad}")
        else:
            print(f"Epoch {epoch+1}: No gradient for strides.")

        if adstft.win_pow.grad is not None:
            print(f"Epoch {epoch+1}: Gradient for window power: {adstft.win_pow.grad}")
        else:
            print(f"Epoch {epoch+1}: No gradient for window power.")
        """

    # 9. Final Gradient Verification after Training
    print("\nFinal Gradient Verification:")
    if dstft.win_length.grad is not None:
        print(f"Gradient for window length: {dstft.win_length.grad}")
    else:
        print("No gradient for window length.")

    if dstft.strides.grad is not None:
        print(f"Gradient for strides: {dstft.strides.grad}")
    else:
        print("No gradient for strides.")

    if dstft.win_pow.grad is not None:
        print(f"Gradient for window power: {dstft.win_pow.grad}")
    else:
        print("No gradient for window power.")


if __name__ == "__main__":
    main()
