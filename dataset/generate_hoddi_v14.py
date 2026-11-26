"""
Generate HODDI_v14 dataset with 1:X negative sampling ratio.

For each positive sample with N drugs:
- Generate N negative samples by replacing one drug at a time
- Generate 1 negative sample by replacing the SE
- Total: N+1 negative samples per positive sample

Then remove all Pos ⊂ Neg conflict samples.
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
    """Load all positive samples from HODDI_v6 (original)."""
    pos_dir = os.path.join(input_dir, 'HODDI', 'HGNN', 'positive_samples')

    samples = []
    pos_pairs = set()
    all_drugs = set()
    all_ses = set()

    for filename in sorted(os.listdir(pos_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(pos_dir, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                drugs = ast.literal_eval(row['DrugBankID'])
                se = row['SE_above_0.9']

                all_drugs.update(drugs)
                all_ses.add(se)

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

    return samples, pos_pairs, sorted(all_drugs), sorted(all_ses)


def generate_negatives_1_to_x(pos_samples, pos_pairs, all_drugs, all_ses, random_state=42):
    """
    Generate negative samples with 1:X ratio.
    For N-drug positive: generate N (replace each drug) + 1 (replace SE) = N+1 negatives.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    neg_samples = []
    neg_pairs = set()

    # Build drug frequency for smart replacement
    drug_freq = Counter()
    for s in pos_samples:
        drug_freq.update(s['drugs'])

    # Group SE by frequency for smart replacement
    se_freq = Counter([s['se'] for s in pos_samples])
    se_list = list(all_ses)

    stats = {
        'total_pos': len(pos_samples),
        'neg_by_drug_replace': 0,
        'neg_by_se_replace': 0,
        'skipped_overlap': 0,
        'skipped_duplicate': 0,
    }

    print("Generating negative samples (1:X ratio)...")
    for pos in tqdm(pos_samples):
        drugs = pos['drugs']
        se = pos['se']
        n_drugs = len(drugs)

        # Strategy 1: Replace each drug one at a time
        for i in range(n_drugs):
            # Try to find a valid replacement drug
            found = False
            attempts = 0
            max_attempts = 100

            while not found and attempts < max_attempts:
                attempts += 1
                # Pick a random drug not in current combo
                replacement = random.choice(all_drugs)
                if replacement in drugs:
                    continue

                # Create new drug combo
                new_drugs = drugs.copy()
                new_drugs[i] = replacement
                new_drugs_sorted = sorted(new_drugs)
                new_combo = str(new_drugs_sorted)

                pair = (new_combo, se)

                if pair in pos_pairs:
                    stats['skipped_overlap'] += 1
                    continue
                if pair in neg_pairs:
                    stats['skipped_duplicate'] += 1
                    continue

                # Valid negative sample
                neg_samples.append({
                    'drug_combo': new_combo,
                    'drugs': new_drugs_sorted,
                    'drug_set': frozenset(new_drugs_sorted),
                    'se': se,
                    'drug_count': n_drugs,
                    'neg_type': 'drug_replace',
                })
                neg_pairs.add(pair)
                stats['neg_by_drug_replace'] += 1
                found = True

        # Strategy 2: Replace SE
        found = False
        attempts = 0
        max_attempts = 100

        while not found and attempts < max_attempts:
            attempts += 1
            new_se = random.choice(se_list)
            if new_se == se:
                continue

            pair = (pos['drug_combo'], new_se)

            if pair in pos_pairs:
                stats['skipped_overlap'] += 1
                continue
            if pair in neg_pairs:
                stats['skipped_duplicate'] += 1
                continue

            neg_samples.append({
                'drug_combo': pos['drug_combo'],
                'drugs': drugs,
                'drug_set': pos['drug_set'],
                'se': new_se,
                'drug_count': n_drugs,
                'neg_type': 'se_replace',
            })
            neg_pairs.add(pair)
            stats['neg_by_se_replace'] += 1
            found = True

    return neg_samples, stats


def find_and_remove_conflicts(pos_samples, neg_samples):
    """
    Find and remove all Pos ⊂ Neg conflicts.
    Returns clean samples without any conflicts.
    """
    # Group by SE for efficiency
    se_to_pos = defaultdict(list)
    se_to_neg = defaultdict(list)

    for i, s in enumerate(pos_samples):
        se_to_pos[s['se']].append((i, s['drug_set'], s['drug_count']))

    for i, s in enumerate(neg_samples):
        se_to_neg[s['se']].append((i, s['drug_set'], s['drug_count']))

    pos_in_conflict = set()
    neg_in_conflict = set()
    conflict_count = 0

    print("Finding Pos ⊂ Neg conflicts...")
    for se in tqdm(se_to_pos.keys()):
        if se not in se_to_neg:
            continue

        for pos_idx, pos_drugs, pos_count in se_to_pos[se]:
            for neg_idx, neg_drugs, neg_count in se_to_neg[se]:
                if pos_count < neg_count and pos_drugs < neg_drugs:
                    pos_in_conflict.add(pos_idx)
                    neg_in_conflict.add(neg_idx)
                    conflict_count += 1

    # Filter out conflict samples
    clean_pos = [s for i, s in enumerate(pos_samples) if i not in pos_in_conflict]
    clean_neg = [s for i, s in enumerate(neg_samples) if i not in neg_in_conflict]

    return clean_pos, clean_neg, len(pos_in_conflict), len(neg_in_conflict), conflict_count


def generate_hoddi_v14(input_dir, output_dir, random_state=42):
    """Generate HODDI_v14 with 1:X ratio and no conflicts."""
    print("=" * 70)
    print("Generating HODDI_v14 Dataset")
    print("(1:X negative ratio + Remove Pos ⊂ Neg conflicts)")
    print("=" * 70)
    print("\nStrategy:")
    print("  - For N-drug positive: generate N+1 negatives")
    print("    - N negatives: replace one drug at a time")
    print("    - 1 negative: replace SE")
    print("  - Remove all Pos ⊂ Neg conflict samples")
    print("=" * 70)

    # Create output directories
    hgnn_output_dir = os.path.join(output_dir, 'HODDI', 'HGNN')
    pos_output_dir = os.path.join(hgnn_output_dir, 'positive_samples')
    neg_output_dir = os.path.join(hgnn_output_dir, 'negative_samples')

    os.makedirs(pos_output_dir, exist_ok=True)
    os.makedirs(neg_output_dir, exist_ok=True)

    # Copy dictionary folder
    dict_input_dir = os.path.join(input_dir, 'dictionary')
    dict_output_dir = os.path.join(output_dir, 'dictionary')
    if os.path.exists(dict_input_dir):
        if os.path.exists(dict_output_dir):
            shutil.rmtree(dict_output_dir)
        shutil.copytree(dict_input_dir, dict_output_dir)
        print(f"Copied dictionary folder")

    # Step 1: Load positive samples
    print("\nStep 1: Loading positive samples...")
    pos_samples, pos_pairs, all_drugs, all_ses = load_positive_samples(input_dir)
    print(f"  Positive samples: {len(pos_samples):,}")
    print(f"  Unique drugs: {len(all_drugs):,}")
    print(f"  Unique SEs: {len(all_ses):,}")

    # Step 2: Generate negatives with 1:X ratio
    print("\nStep 2: Generating negatives (1:X ratio)...")
    neg_samples, gen_stats = generate_negatives_1_to_x(
        pos_samples, pos_pairs, all_drugs, all_ses, random_state
    )

    print(f"\n  Generated negative samples: {len(neg_samples):,}")
    print(f"    - By drug replacement: {gen_stats['neg_by_drug_replace']:,}")
    print(f"    - By SE replacement: {gen_stats['neg_by_se_replace']:,}")
    print(f"  Skipped (overlap with pos): {gen_stats['skipped_overlap']:,}")
    print(f"  Skipped (duplicate): {gen_stats['skipped_duplicate']:,}")
    print(f"  Ratio (neg/pos): {len(neg_samples)/len(pos_samples):.2f}:1")

    # Step 3: Remove Pos ⊂ Neg conflicts
    print("\nStep 3: Removing Pos ⊂ Neg conflicts...")
    clean_pos, clean_neg, pos_removed, neg_removed, conflict_pairs = find_and_remove_conflicts(
        pos_samples, neg_samples
    )

    print(f"  Conflict pairs found: {conflict_pairs:,}")
    print(f"  Positive samples removed: {pos_removed:,} ({100*pos_removed/len(pos_samples):.2f}%)")
    print(f"  Negative samples removed: {neg_removed:,} ({100*neg_removed/len(neg_samples):.2f}%)")
    print(f"  Clean positive samples: {len(clean_pos):,}")
    print(f"  Clean negative samples: {len(clean_neg):,}")

    # Step 4: Verify no conflicts remain
    print("\nStep 4: Verifying no conflicts remain...")
    _, _, _, _, remaining = find_and_remove_conflicts(clean_pos, clean_neg)
    print(f"  Remaining conflicts: {remaining}")
    assert remaining == 0, "Still have conflicts!"
    print("  ✓ No Pos ⊂ Neg conflicts in v14!")

    # Step 5: Analyze distribution
    print("\nStep 5: Distribution by drug count...")

    pos_by_count = defaultdict(list)
    neg_by_count = defaultdict(list)

    for s in clean_pos:
        pos_by_count[s['drug_count']].append(s)
    for s in clean_neg:
        neg_by_count[s['drug_count']].append(s)

    print(f"\n{'Order':<10} {'Pos':>10} {'Neg':>10} {'Ratio':>8} {'Expected':>10}")
    print("-" * 55)

    for dc in sorted(set(pos_by_count.keys()) | set(neg_by_count.keys())):
        if dc > 10:
            continue
        n_pos = len(pos_by_count[dc])
        n_neg = len(neg_by_count[dc])
        ratio = n_neg / n_pos if n_pos > 0 else 0
        expected = dc + 1  # N drugs + 1 SE replacement
        print(f"{dc}-drug     {n_pos:>10,} {n_neg:>10,} {ratio:>7.2f}x  ({expected}x expected)")

    # Step 6: Write output files
    print("\nStep 6: Writing output files...")

    # Write positive samples
    for dc in sorted(pos_by_count.keys()):
        samples = pos_by_count[dc]
        if not samples:
            continue

        rows = [s['row'] for s in samples]
        df = pd.DataFrame(rows)
        filename = f"{dc}_drug.csv"
        df.to_csv(os.path.join(pos_output_dir, filename), index=False)

    # Write negative samples
    for dc in sorted(neg_by_count.keys()):
        samples = neg_by_count[dc]
        if not samples:
            continue

        rows = []
        for i, s in enumerate(samples):
            row = {
                'report_id': f"neg_{dc}_{i}",
                'DrugBankID': s['drug_combo'],
                'SE_above_0.9': s['se'],
                'hyperedge_label': 0,
                'drug_count': dc,
                'neg_type': s['neg_type'],
                'prr': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
                'reports_count': 0
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        filename = f"{dc}_drug.csv"
        df.to_csv(os.path.join(neg_output_dir, filename), index=False)

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("HODDI_v14 Generation Complete!")
    print("=" * 70)

    total = len(clean_pos) + len(clean_neg)
    print(f"\nSummary:")
    print(f"  Total samples: {total:,}")
    print(f"  Positive: {len(clean_pos):,} ({100*len(clean_pos)/total:.1f}%)")
    print(f"  Negative: {len(clean_neg):,} ({100*len(clean_neg)/total:.1f}%)")
    print(f"  Neg/Pos ratio: {len(clean_neg)/len(clean_pos):.2f}:1")
    print(f"\nOutput: {output_dir}")

    # Save report
    report = {
        'strategy': '1:X negative ratio (N drugs + 1 SE replacement) + Remove Pos⊂Neg conflicts',
        'original_positive': len(pos_samples),
        'generated_negative': len(neg_samples),
        'neg_by_drug_replace': gen_stats['neg_by_drug_replace'],
        'neg_by_se_replace': gen_stats['neg_by_se_replace'],
        'conflict_pairs': conflict_pairs,
        'pos_removed': pos_removed,
        'neg_removed': neg_removed,
        'final_positive': len(clean_pos),
        'final_negative': len(clean_neg),
        'final_total': total,
        'neg_pos_ratio': len(clean_neg) / len(clean_pos),
    }

    with open(os.path.join(output_dir, 'generation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset/HODDI_v6',
                        help='Input HODDI directory (with original positive samples)')
    parser.add_argument('--output', type=str, default='dataset/HODDI_v14',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.isabs(args.input):
        args.input = os.path.join(PROJECT_ROOT, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(PROJECT_ROOT, args.output)

    generate_hoddi_v14(args.input, args.output, args.seed)
