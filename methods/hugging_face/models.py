import torch
import torch.nn as nn
from transformers import AutoModel

class CNN8RNN(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        """Classifier for a new task using pretrained CNN8RNN as a sub-module."""
        super(CNN8RNN, self).__init__()
        
        # Step 1: Load the pretrained CNN8RNN model
        self.base = AutoModel.from_pretrained(
            "wsntxxn/cnn8rnn-audioset-sed", trust_remote_code=True
        )
        
        # Optional: Freeze the base model to prevent its weights from being updated
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Step 2: Transfer to another task layer (define custom head)
        # Assuming 512 is the size of the output embeddings from the base model.
        # You can check this in the documentation or experiment to get the correct size.
        self.fc_transfer = nn.Linear(447, num_classes, bias=True)

        # Step 3: Initialize weights of the new fully connected layer
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the custom classifier head."""
        nn.init.xavier_uniform_(self.fc_transfer.weight)
        if self.fc_transfer.bias is not None:
            nn.init.constant_(self.fc_transfer.bias, 0)

    def load_finetuned_weights(self, checkpoint_path):
        """Load fine-tuned weights into the model."""
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_dict = self.state_dict()

        # Filter out mismatched layers to allow partial loading
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model's state_dict with the pre-trained layers
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Forward pass through the model.
        Input: (batch_size, num_channels, seq_length)
        """
        # Step 4: Forward pass through the base model (CNN8RNN)
        output_dict = self.base(input)

        embedding = output_dict['clipwise_output']

        # Step 5: Forward pass through the custom fully connected layer for classification
        output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        return output
