# File: models/model_selection.py

import torch
from methods.panns.template import PANNS_CNN6, PANNS_RESNET22, PANNS_MOBILENETV1, PANNS_WAVEGRAM_CNN14
from methods.hugging_face.models import CNN8RNN
from methods.ast.models import ASTModel

def get_model(args):
    if args.model_name == 'panns_cnn6':
        model = PANNS_CNN6(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes,
            frontend=args.frontend,
            batch_size=args.batch_size,
        )
        model.load_from_pretrain("./weights/Cnn6_mAP=0.343.pth")
    elif args.model_name == 'panns_resnet22':
        model = PANNS_RESNET22(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_from_pretrain("./weights/ResNet22_mAP=0.430.pth") 
    elif args.model_name == 'panns_mobilenetv1':
        model = PANNS_MOBILENETV1(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_from_pretrain("./weights/MobileNetV1_mAP=0.389.pth") 
    elif args.model_name == 'panns_wavegram_cnn14':
        model = PANNS_WAVEGRAM_CNN14(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_from_pretrain("./weights/Wavegram_Cnn14_mAP=0.389.pth") 
    elif args.model_name == 'cnn8rnn':
        model = CNN8RNN(
            num_classes=args.num_classes
        )
    if args.model_name == 'ast':
        model = ASTModel(
            label_dim=args.num_classes,
            input_tdim=251
        )
    else: 
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
