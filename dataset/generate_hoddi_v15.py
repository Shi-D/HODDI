"""
Generate HODDI_v15 dataset with distribution-based negative sampling.

For each positive sample with N drugs:
- Generate N negative samples by replacing each drug with a drug sampled
  from the positive samples' drug distribution
- Generate 1 negative sample by replacing SE with an SE sampled from
  the positive samples' SE distribution
- Total: N+1 negative samples per positive sample

This maintains 1:X ratio but uses the positive distribution for sampling.
"""

import os
import sys
import ast
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import random
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def load_positive_samples(input_dir):
    """Load all positive samples from HODDI_v14."""
    pos_dir = os.path.join(input_dir, 'HODDI', 'HGNN', 'positive_samples')

    samples = []
    pos_pairs = set()
    drug_counts = Counter()
    se_counts = Counter()

    for filename in sorted(os.listdir(pos_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(pos_dir, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                drugs = ast.literal_eval(row['DrugBankID'])
                se = row['SE_above_0.9']

                drug_counts.update(drugs)
                se_counts[se] += 1

                pair = (row['DrugBankID'], se)
                pos_pairs.add(pair)

                samples.append({
                    'drug_combo': row['DrugBankID'],
                    'drugs': drugs,
                    'drug_set': frozenset(drugs),
                    'se': se,
                    'drug_count': len(drugs),
                    'row': row.to_dict(),
                })

    return samples, pos_pairs, drug_counts, se_counts


def generate_negatives_distribution_based(pos_samples, pos_pairs, drug_counts, se_counts, random_state=42):
    """
    Generate negative samples using distribution-based sampling.
    For N-drug positive: generate N (replace each drug by distribution) + 1 (replace SE by distribution) = N+1 negatives.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    neg_samples = []
    neg_pairs = set()

    # Build drug sampling distribution from positive samples
    drug_list = list(drug_counts.keys())
    drug_probs = np.array([drug_counts[d] for d in drug_list], dtype=float)
    drug_probs /= drug_probs.sum()

    # Build SE sampling distribution from positive samples
    se_list = list(se_counts.keys())
    se_probs = np.array([se_counts[se] for se in se_list], dtype=float)
    se_probs /= se_probs.sum()

    stats = {
        'total_pos': len(pos_samples),
        'neg_by_drug_replace': 0,
        'neg_by_se_replace': 0,
        'skipped_same_drug': 0,
        'skipped_same_se': 0,
        'skipped_duplicate': 0,
        'skipped_in_pos': 0,
    }

    print("Generating negative samples (distribution-based 1:X)...")
    for pos in tqdm(pos_samples):
        drugs = pos['drugs']
        se = pos['se']
        n_drugs = len(drugs)

        # Strategy 1: Replace each drug one at a time (sample from drug distribution)
        for i in range(n_drugs):
            found = False
            attempts = 0
            max_attempts = 50

            while not found and attempts < max_attempts:
                # Sample replacement drug from positive distribution
                new_drug = np.random.choice(drug_list, p=drug_probs)

                if new_drug == drugs[i]:
                    stats['skipped_same_drug'] += 1
                    attempts += 1
                    continue

                if new_drug in drugs:
                    # Already in the drug combination
                    attempts += 1
                    continue

                # Create new drug combination
                new_drugs = drugs[:i] + [new_drug] + drugs[i+1:]
                new_drug_combo = str(new_drugs)
                pair = (new_drug_combo, se)

                if pair in pos_pairs:
                    stats['skipped_in_pos'] += 1
                    attempts += 1
                    continue

                if pair in neg_pairs:
                    stats['skipped_duplicate'] += 1
                    attempts += 1
                    continue

                # Valid negative sample
                neg_pairs.add(pair)
                new_row = pos['row'].copy()
                new_row['DrugBankID'] = new_drug_combo
                new_row['hyperedge_label'] = 0

                neg_samples.append({
                    'drug_combo': new_drug_combo,
                    'drugs': new_drugs,
                    'drug_set': frozenset(new_drugs),
                    'se': se,
                    'drug_count': len(new_drugs),
                    'row': new_row,
                })
                stats['neg_by_drug_replace'] += 1
                found = True

        # Strategy 2: Replace SE (sample from SE distribution)
        found = False
        attempts = 0
        max_attempts = 50

        while not found and attempts < max_attempts:
            new_se = np.random.choice(se_list, p=se_probs)

            if new_se == se:
                stats['skipped_same_se'] += 1
                attempts += 1
                continue

            pair = (pos['drug_combo'], new_se)

            if pair in pos_pairs:
                stats['skipped_in_pos'] += 1
                attempts += 1
                continue

            if pair in neg_pairs:
                stats['skipped_duplicate'] += 1
                attempts += 1
                continue

            # Valid negative sample
            neg_pairs.add(pair)
            new_row = pos['row'].copy()
            new_row['SE_above_0.9'] = new_se
            new_row['hyperedge_label'] = 0

            neg_samples.append({
                'drug_combo': pos['drug_combo'],
                'drugs': pos['drugs'],
                'drug_set': pos['drug_set'],
                'se': new_se,
                'drug_count': pos['drug_count'],
                'row': new_row,
            })
            stats['neg_by_se_replace'] += 1
            found = True

    print(f"\nGeneration statistics:")
    print(f"  Total positive samples: {stats['total_pos']}")
    print(f"  Neg by drug replacement: {stats['neg_by_drug_replace']}")
    print(f"  Neg by SE replacement: {stats['neg_by_se_replace']}")
    print(f"  Total negative samples: {len(neg_samples)}")
    print(f"  Ratio (neg/pos): {len(neg_samples)/stats['total_pos']:.2f}")

    return neg_samples, stats


def save_samples(samples, output_dir, sample_type):
    """Save samples grouped by drug count."""
    os.makedirs(output_dir, exist_ok=True)

    by_count = defaultdict(list)
    for s in samples:
        by_count[s['drug_count']].append(s['row'])

    for count, rows in sorted(by_count.items()):
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, f'{count}_drug.csv')
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(df)} {sample_type} samples with {count} drugs")


def main():
    input_dir = os.path.join(PROJECT_ROOT, 'dataset', 'HODDI_v14')
    output_dir = os.path.join(PROJECT_ROOT, 'dataset', 'HODDI_v15')

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Load positive samples
    print("\nLoading positive samples...")
    pos_samples, pos_pairs, drug_counts, se_counts = load_positive_samples(input_dir)
    print(f"Loaded {len(pos_samples)} positive samples")
    print(f"Unique drugs: {len(drug_counts)}")
    print(f"Unique SEs: {len(se_counts)}")

    # Generate negative samples
    neg_samples, stats = generate_negatives_distribution_based(
        pos_samples, pos_pairs, drug_counts, se_counts, random_state=42
    )

    # Create output directory structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Copy dictionary
    src_dict = os.path.join(input_dir, 'dictionary')
    dst_dict = os.path.join(output_dir, 'dictionary')
    shutil.copytree(src_dict, dst_dict)
    print(f"\nCopied dictionary to {dst_dict}")

    # Save HGNN format
    hgnn_dir = os.path.join(output_dir, 'HODDI', 'HGNN')
    pos_out_dir = os.path.join(hgnn_dir, 'positive_samples')
    neg_out_dir = os.path.join(hgnn_dir, 'negative_samples')

    print("\nSaving positive samples...")
    save_samples(pos_samples, pos_out_dir, 'positive')

    print("\nSaving negative samples...")
    save_samples(neg_samples, neg_out_dir, 'negative')

    # Verify SE balance
    print("\n=== Verifying SE distribution ===")
    pos_se_counts = Counter([s['se'] for s in pos_samples])
    neg_se_counts = Counter([s['se'] for s in neg_samples])

    all_se = set(pos_se_counts.keys()) | set(neg_se_counts.keys())
    ratios = []
    for se in all_se:
        pos = pos_se_counts.get(se, 0)
        neg = neg_se_counts.get(se, 0)
        if pos > 0:
            ratios.append(neg / pos)

    print(f"Total SEs: {len(all_se)}")
    print(f"Neg/Pos ratio per SE - Min: {np.min(ratios):.2f}, Max: {np.max(ratios):.2f}, "
          f"Mean: {np.mean(ratios):.2f}, Median: {np.median(ratios):.2f}")

    # Save stats
    stats_file = os.path.join(output_dir, 'generation_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'total_pos': len(pos_samples),
            'total_neg': len(neg_samples),
            'neg_pos_ratio': len(neg_samples) / len(pos_samples),
            'unique_drugs': len(drug_counts),
            'unique_ses': len(se_counts),
            'neg_pos_ratio_per_se_mean': float(np.mean(ratios)),
            'neg_pos_ratio_per_se_median': float(np.median(ratios)),
            'generation_stats': stats,
        }, f, indent=2)

    print(f"\nStats saved to {stats_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
