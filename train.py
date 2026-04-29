import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
from torchsummary import summary
# Import data loader
from dataloaders import get_data_loader, DATASET_CONFIGS


def import_models(model_type='naive'):
    """
    Dynamically import model module based on model type

    Args:
        model_type: Model type, optional 'naive', 'mobile', 'pann'
            - naive: use models.py
            - mobile: use models_mobilenet.py
            - pann: use models_pann.py

    Returns:
        Imported model classes
    """
    if model_type == 'naive':
        from models_naive import (
            MelCNN, MelLDSCNN,
            MelHilbertCNN, MelHilbertLDSCNN,
            MelHilbertTimeCNN, MelHilbertTimeLDSCNN,
            SignalHilbertCNN, SignalHilbertLDSCNN,
            SignalCNN, SignalLDSCNN,
        )
    elif model_type == 'mobile':
        from models_mobilenet import (
            MelCNN, MelLDSCNN,
            MelHilbertCNN, MelHilbertLDSCNN,
            MelHilbertTimeCNN, MelHilbertTimeLDSCNN,
            SignalHilbertCNN, SignalHilbertLDSCNN,
            SignalCNN, SignalLDSCNN,
        )
    elif model_type == 'pann':
        from models_pann import (
            MelCNN, MelLDSCNN,
            MelHilbertCNN, MelHilbertLDSCNN,
            MelHilbertTimeCNN, MelHilbertTimeLDSCNN,
            SignalHilbertCNN, SignalHilbertLDSCNN,
            SignalCNN, SignalLDSCNN,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: naive, mobile, pann")
    
    return (MelCNN, MelLDSCNN,
            MelHilbertCNN, MelHilbertLDSCNN,
            MelHilbertTimeCNN, MelHilbertTimeLDSCNN,
            SignalHilbertCNN, SignalHilbertLDSCNN,
            SignalCNN, SignalLDSCNN)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_model(type, model_backend='naive', num_classes=3, n_mels=128, fixed_length=128, hilbert_height=128, hilbert_width=128, **kwargs):
    """Get the corresponding model based on type"""
    # Dynamically import models
    (MelCNN, MelLDSCNN,
     MelHilbertCNN, MelHilbertLDSCNN,
     MelHilbertTimeCNN, MelHilbertTimeLDSCNN,
     SignalHilbertCNN, SignalHilbertLDSCNN,
     SignalCNN, SignalLDSCNN) = import_models(model_backend)
    
    if type == 'Mel':
        return MelCNN(num_classes=num_classes, n_mels=n_mels, time_frames=fixed_length)
    elif type == 'MelLDS':
        return MelLDSCNN(num_classes=num_classes, n_mels=n_mels, time_frames=fixed_length)
    elif type == 'MelHilbert':
        return MelHilbertCNN(num_classes=num_classes, in_channels=128)
    elif type == 'MelHilbertLDS':
        return MelHilbertLDSCNN(num_classes=num_classes, in_channels=128)
    elif type == 'MelHilbertTime':
        return MelHilbertTimeCNN(num_classes=num_classes, in_channels=128)
    elif type == 'MelHilbertTimeLDS':
        return MelHilbertTimeLDSCNN(num_classes=num_classes, in_channels=128)
    elif type == 'SignalHilbert':
        return SignalHilbertCNN(num_classes=num_classes, hilbert_height=hilbert_height, hilbert_width=hilbert_width)
    elif type == 'SignalHilbertLDS':
        return SignalHilbertLDSCNN(num_classes=num_classes, hilbert_height=hilbert_height, hilbert_width=hilbert_width)
    elif type == 'Signal':
        return SignalCNN(num_classes=num_classes, signal_length=1024)
    elif type == 'SignalLDS':
        return SignalLDSCNN(num_classes=num_classes, signal_length=1024)
    else:
        raise ValueError(f"Unsupported type: {type}")

def train(model, train_loader, val_loader, device, writer, model_path, categories, num_epochs=100):
    """
    Train model

    Args:
        model: Model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        writer: TensorBoard writer
        model_path: Model save path
        categories: Category list
        num_epochs: Number of training epochs
    """
    num_classes = len(categories)

    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_acc = 0
    best_f1 = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Create counters for each class
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        # Create confusion matrix for computing precision and recall
        train_confusion_matrix = torch.zeros(num_classes, num_classes)
        
        # Use tqdm to show progress
        # pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        # for inputs, labels in pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Compute accuracy for each class
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                
                # Update confusion matrix
                train_confusion_matrix[label][pred] += 1
            
            # Update progress bar info
            # pbar.set_postfix({'acc': f'{100. * correct / total:.2f}%'})
        
        # Compute training accuracy
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Compute training accuracy for each class
        train_class_acc = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(class_total))]
        
        # Compute training precision, recall and F1
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        train_class_precision = [0] * num_classes
        train_class_recall = [0] * num_classes
        
        for i in range(num_classes):
            # Precision: proportion of actual class i among samples predicted as class i
            if train_confusion_matrix[:, i].sum().item() > 0:
                train_class_precision[i] = train_confusion_matrix[i, i].item() / train_confusion_matrix[:, i].sum().item()
            
            # Recall: proportion of correctly predicted samples among actual class i
            if train_confusion_matrix[i, :].sum().item() > 0:
                train_class_recall[i] = train_confusion_matrix[i, i].item() / train_confusion_matrix[i, :].sum().item()
        
        # Compute macro-average precision, recall and F1
        train_precision = sum(train_class_precision) / num_classes
        train_recall = sum(train_class_recall) / num_classes
        
        if train_precision + train_recall > 0:
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Create counters for each class
        val_class_correct = [0] * num_classes
        val_class_total = [0] * num_classes
        
        # Create confusion matrix for computing precision and recall
        val_confusion_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Compute accuracy for each class and update confusion matrix
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if label < len(val_class_total):
                        val_class_total[label] += 1
                        if pred == label:
                            val_class_correct[label] += 1
                    
                    # Update confusion matrix
                    val_confusion_matrix[label][pred] += 1
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        # Compute validation accuracy for each class
        val_class_acc = [100. * val_class_correct[i] / val_class_total[i] if val_class_total[i] > 0 else 0 for i in range(len(val_class_total))]
        
        # Compute validation precision, recall and F1
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        val_class_precision = [0] * num_classes
        val_class_recall = [0] * num_classes
        
        for i in range(num_classes):
            # Precision: proportion of actual class i among samples predicted as class i
            if val_confusion_matrix[:, i].sum().item() > 0:
                val_class_precision[i] = val_confusion_matrix[i, i].item() / val_confusion_matrix[:, i].sum().item()
            
            # Recall: proportion of correctly predicted samples among actual class i
            if val_confusion_matrix[i, :].sum().item() > 0:
                val_class_recall[i] = val_confusion_matrix[i, i].item() / val_confusion_matrix[i, :].sum().item()
        
        # Compute macro-average precision, recall and F1
        val_precision = sum(val_class_precision) / num_classes
        val_recall = sum(val_class_recall) / num_classes
        
        if val_precision + val_recall > 0:
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
        
        # Update learning rate
        scheduler.step()
        
        # Record to tensorboard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalars('Precision', {'train': train_precision, 'val': val_precision}, epoch)
        writer.add_scalars('Recall', {'train': train_recall, 'val': val_recall}, epoch)
        writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch)
        
        # Record accuracy for each class
        for i, class_name in enumerate(categories):
            writer.add_scalars(f'Class_{class_name}_Accuracy', {'train': train_class_acc[i], 'val': val_class_acc[i]}, epoch)
        
        # # Print training info
        # print(f'Epoch {epoch+1}/{num_epochs}:')
        # print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}')
        # print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
        # print('-' * 60)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), model_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)

    writer.close()
    print(f'Best validation accuracy: {best_acc:.2f}%   Best validation F1: {best_f1:.4f}')


def get_model_params(type, hop_length=128):
    """Return corresponding model parameters based on type"""
    if type == 'Mel':
        return {
            'n_mels': 128,
            'fixed_length': 128
        }
    elif type == 'MelLDS':
        return {
            'n_mels': 128,
            'fixed_length': 128,
            'hop_length': hop_length
        }
    elif type == 'MelHilbert':
        return {
            'n_mels': 128,
            'mel_time_frames': 128,
            'hilbert_height': 8,
            'hilbert_width': 16,
            'in_channels': 128
        }
    elif type == 'MelHilbertLDS':
        return {
            'n_mels': 128,
            'mel_time_frames': 128,
            'hilbert_height': 8,
            'hilbert_width': 16,
            'in_channels': 128,
            'hop_length': hop_length
        }
    elif type == 'MelHilbertTime':
        return {
            'n_mels': 128,
            'mel_time_frames': 128,
            'hilbert_height': 8,
            'hilbert_width': 16,
            'in_channels': 128
        }
    elif type == 'MelHilbertTimeLDS':
        return {
            'n_mels': 128,
            'mel_time_frames': 128,
            'hilbert_height': 8,
            'hilbert_width': 16,
            'in_channels': 128,
            'hop_length': hop_length
        }
    elif type == 'SignalHilbert':
        return {
            'hilbert_height': 128,
            'hilbert_width': 128
        }
    elif type == 'SignalHilbertLDS':
        return {
            'hilbert_height': 128,
            'hilbert_width': 128
        }
    elif type == 'Signal':
        return {
            'signal_length': 1024
        }
    elif type == 'SignalLDS':
        return {
            'signal_length': 1024
        }
    else:
        raise ValueError(f"Unsupported type: {type}")

def main():
    parser = argparse.ArgumentParser(description='Train CSS model')
    parser.add_argument('--type', type=str, required=True,
                      help='Model type: Mel, MelLDS, '
                           'MelHilbert, MelHilbertLDS, '
                           'MelHilbertTime, MelHilbertTimeLDS, '
                           'SignalHilbert, SignalHilbertLDS, '
                           'Signal, SignalLDS')
    parser.add_argument('--date', type=str, required=True,
                      help='Experiment date, used for tensorboard logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--hop_length', type=int, default=64,
                      help='Hop length')
    parser.add_argument('--dataset', type=str, default='cough-speech-sneeze',
                      help='Dataset name: cough-speech-sneeze, CoughDataset or ESC50-human')
    parser.add_argument('--model_backend', type=str, default='naive',
                      help='Model backend: naive (models.py), mobile (models_mobilenet.py), pann (models_pann.py)')
    args = parser.parse_args()

    print('#'*50)
    print(f'{args.type}\nseed: {args.seed}\ndate: {args.date}\nhop_length: {args.hop_length}\ndataset: {args.dataset}\nmodel_backend: {args.model_backend}')
    print('#'*50)
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {list(DATASET_CONFIGS.keys())}")
    
    dataset_config = DATASET_CONFIGS[args.dataset]
    categories = dataset_config['categories']
    num_classes = len(categories)
    print(f'Dataset: {args.dataset}, Num classes: {num_classes}, Categories: {categories}')
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tensorboard log directory, with LDS hop_length annotation and dataset name
    if 'LDS' in args.type:
        run_name = f'css_{args.dataset}_{args.model_backend}_{args.type}_{args.hop_length}_{args.seed}'
    else:
        run_name = f'css_{args.dataset}_{args.model_backend}_{args.type}_{args.seed}'
    log_dir = f'runs_{args.date}/{run_name}'
    writer = SummaryWriter(log_dir)
    
    # Get model parameters
    model_params = get_model_params(args.type, args.hop_length)
    
    # Get data loader
    train_loader = get_data_loader(
        mode='train',
        type=args.type,
        dataset_name=args.dataset,
        data_augmentation=True,
        seed=args.seed,
        num_workers=8,
        **model_params
    )
    
    val_loader = get_data_loader(
        mode='val',
        type=args.type,
        dataset_name=args.dataset,
        data_augmentation=False,
        seed=args.seed,
        num_workers=8,
        **model_params
    )
    
    # Get model (pass in num_classes and model backend)
    model = get_model(args.type, model_backend=args.model_backend, num_classes=num_classes, **model_params).to(device)
    
    # # summary, determine input dimensions based on type
    # if args.type in ['Mel', 'MelLDS']:
    #     summary(model, (1, 128, 128))
    # elif args.type in ['MelHilbert', 'MelHilbertLDS']:
    #     summary(model, (128, 8, 16))
    # elif args.type in ['MelHilbertTime', 'MelHilbertTimeLDS']:
    #     summary(model, (128, 8, 16))
    # elif args.type in ['SignalHilbert', 'SignalHilbertLDS']:
    #     summary(model, (1, 128, 128))
    # elif args.type in ['Signal', 'SignalLDS']:
    #     summary(model, (1, 1024))
    
    # Create checkpoints directory
    os.makedirs(f'checkpoints_{args.date}', exist_ok=True)
    model_path = f'checkpoints_{args.date}/{run_name}.pth'
    
    # Train model
    train(model, train_loader, val_loader, device, writer, model_path, categories, num_epochs=100)
    
    # Close tensorboard writer
    writer.close()

if __name__ == '__main__':
    main() 