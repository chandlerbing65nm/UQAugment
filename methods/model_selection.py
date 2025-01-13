# File: models/model_selection.py

import torch
from methods.panns.template import PANNS_CNN6, PANNS_CNN14, PANNS_RESNET22, PANNS_MOBILENETV1, PANNS_MOBILENETV2, PANNS_WAVEGRAM_CNN14
from methods.hugging_face.models import CNN8RNN
from methods.ast.template import AudioSpectrogramTransformer

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
            args=args
        )
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/Cnn6_mAP=0.343.pth")
    elif args.model_name == 'panns_cnn14':
        model = PANNS_CNN14(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes,
            frontend=args.frontend,
            batch_size=args.batch_size,
            args=args
        )
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/Cnn14_mAP=0.431.pth")
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
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/ResNet22_mAP=0.430.pth") 
    elif args.model_name == 'panns_mobilenetv1':
        model = PANNS_MOBILENETV1(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes,
            frontend=args.frontend,
            batch_size=args.batch_size,
            args=args
        )
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/MobileNetV1_mAP=0.389.pth") 
    elif args.model_name == 'panns_mobilenetv2':
        model = PANNS_MOBILENETV1(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes,
            frontend=args.frontend,
            batch_size=args.batch_size,
            args=args
        )
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/MobileNetV2_mAP=0.383.pth") 
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
        model.load_from_pretrain("/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/weights/Wavegram_Cnn14_mAP=0.389.pth") 
    elif args.model_name == 'cnn8rnn':
        model = CNN8RNN(
            num_classes=args.num_classes
        )
    elif args.model_name == 'ast':
        model = AudioSpectrogramTransformer(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            num_classes=args.num_classes,
            frontend=args.frontend,  # Change as needed
            batch_size=args.batch_size,
            freeze_base=False,
            device=None,
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size='base384',
            args=args
        )
    else: 
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
