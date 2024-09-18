# analyze_data_statistics.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets.uffia import get_dataloader as uffia_dataloader
from datasets.affia3k import get_dataloader as affia3k_dataloader
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Data Statistics')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=25, help='Random seed')
    parser.add_argument('--classes_num', type=int, default=4, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def check_class_distribution(data_loader, classes_num):
    class_counts = np.zeros(classes_num)
    for batch in data_loader:
        targets = batch['target'].numpy()
        for target in targets:
            class_counts[target.argmax()] += 1
    return class_counts

def save_distribution_plot(class_counts, classes_num, output_path='plots/class_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(range(classes_num), class_counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(classes_num))
    plt.savefig(output_path)
    plt.close()

def save_pie_chart(class_counts, output_path='plots/class_distribution_pie.png'):
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=[f'Class {i}' for i in range(len(class_counts))], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Class Distribution')
    plt.savefig(output_path)
    plt.close()

def save_statistics(class_counts, output_path='plots/class_statistics.txt'):
    total_samples = np.sum(class_counts)
    class_distribution_percentage = (class_counts / total_samples) * 100
    mean = np.mean(class_counts)
    median = np.median(class_counts)
    std_dev = np.std(class_counts)
    max_count = np.max(class_counts)
    min_count = np.min(class_counts)

    with open(output_path, 'w') as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Class distribution (counts): {class_counts}\n")
        f.write(f"Class distribution (percentages): {class_distribution_percentage}\n")
        f.write(f"Mean: {mean}\n")
        f.write(f"Median: {median}\n")
        f.write(f"Standard Deviation: {std_dev}\n")
        f.write(f"Max count: {max_count}\n")
        f.write(f"Min count: {min_count}\n")

def main():
    args = parse_args()

    # Pretty print arguments
    print("Arguments:")
    pprint(vars(args))

    # Set random seed
    np.random.seed(args.seed)

    # Initialize data loader
    _, loader = affia3k_dataloader(split=args.split, batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path)

    # Check class distribution
    class_distribution = check_class_distribution(loader, args.classes_num)
    
    # Save statistical analysis
    save_statistics(class_distribution)

    # Save distribution plot
    save_distribution_plot(class_distribution, args.classes_num)

    # Save pie chart
    save_pie_chart(class_distribution)

    print("Class distribution and statistics saved successfully.")

if __name__ == '__main__':
    main()
