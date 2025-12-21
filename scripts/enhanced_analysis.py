#!/usr/bin/env python3
"""
Enhanced MSR Analysis: Code Complexity, Refactor Patterns, and BERT Sentiment
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load existing data
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

def load_commits():
    """Load commits from CSV"""
    return pd.read_csv(DATA_DIR / "commits.csv", parse_dates=['date'])

def analyze_refactor_temporal_patterns(df):
    """Analyze temporal clustering of refactor/fix commits"""
    print("\n" + "="*60)
    print("REFACTOR TEMPORAL PATTERN ANALYSIS")
    print("="*60)

    # Filter refactor and fix commits
    refactor_keywords = ['refactor', 'fix', 'cleanup', 'clean up', 'simplify', 'improve']

    def is_refactor(msg):
        if pd.isna(msg):
            return False
        msg_lower = msg.lower()
        return any(kw in msg_lower for kw in refactor_keywords)

    df['is_refactor'] = df['message'].apply(is_refactor)
    refactors = df[df['is_refactor']].copy()

    print(f"Total refactor/fix commits: {len(refactors)} ({len(refactors)/len(df)*100:.1f}%)")

    # Sort by date
    refactors = refactors.sort_values('date')

    # Calculate time gaps between consecutive refactors
    refactors['time_gap'] = refactors['date'].diff()

    # Identify clusters (commits within 2 hours of each other)
    cluster_threshold = timedelta(hours=2)
    clusters = []
    current_cluster = []

    for idx, row in refactors.iterrows():
        if pd.isna(row['time_gap']) or row['time_gap'] > cluster_threshold:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [row]
        else:
            current_cluster.append(row)

    if len(current_cluster) >= 2:
        clusters.append(current_cluster)

    # Analyze clusters
    cluster_sizes = [len(c) for c in clusters]
    isolated_refactors = len(refactors) - sum(cluster_sizes)

    print(f"\nReflection-in-action (clustered refactors within 2h):")
    print(f"  - Number of clusters: {len(clusters)}")
    print(f"  - Average cluster size: {np.mean(cluster_sizes):.1f} commits")
    print(f"  - Max cluster size: {max(cluster_sizes) if cluster_sizes else 0}")
    print(f"  - Clustered commits: {sum(cluster_sizes)} ({sum(cluster_sizes)/len(refactors)*100:.1f}%)")

    print(f"\nReflection-on-action (isolated refactors >2h gap):")
    print(f"  - Isolated refactors: {isolated_refactors} ({isolated_refactors/len(refactors)*100:.1f}%)")

    # Find long gaps (>1 week) followed by refactor
    week_gap = timedelta(days=7)
    long_gap_refactors = refactors[refactors['time_gap'] > week_gap]
    print(f"  - Refactors after >1 week gap: {len(long_gap_refactors)}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster size distribution
    if cluster_sizes:
        axes[0].hist(cluster_sizes, bins=range(1, max(cluster_sizes)+2),
                     edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Cluster Size (commits)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Refactor Cluster Size Distribution\n(Reflection-in-Action)')
        axes[0].axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                        label=f'Mean: {np.mean(cluster_sizes):.1f}')
        axes[0].legend()

    # Time gap distribution (log scale for visibility)
    gaps_hours = refactors['time_gap'].dropna().dt.total_seconds() / 3600
    gaps_hours = gaps_hours[gaps_hours > 0]

    axes[1].hist(gaps_hours, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Time Gap (hours)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Time Between Refactor Commits')
    axes[1].axvline(2, color='green', linestyle='--', label='2h threshold')
    axes[1].axvline(168, color='red', linestyle='--', label='1 week')
    axes[1].set_xlim(0, 200)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'refactor_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'refactor_patterns.png'}")

    return {
        'total_refactors': len(refactors),
        'refactor_percentage': len(refactors)/len(df)*100,
        'num_clusters': len(clusters),
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
        'clustered_commits': sum(cluster_sizes),
        'clustered_percentage': sum(cluster_sizes)/len(refactors)*100 if len(refactors) > 0 else 0,
        'isolated_refactors': isolated_refactors,
        'isolated_percentage': isolated_refactors/len(refactors)*100 if len(refactors) > 0 else 0,
        'long_gap_refactors': len(long_gap_refactors)
    }


def run_bert_sentiment_analysis(df):
    """Run BERT sentiment analysis on commit messages"""
    print("\n" + "="*60)
    print("BERT SENTIMENT ANALYSIS")
    print("="*60)

    from transformers import pipeline

    # Use a smaller, faster model for sentiment
    print("Loading BERT sentiment model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )

    # Filter to non-empty messages
    messages = df['message'].dropna().tolist()
    # Truncate long messages (BERT has token limit)
    messages = [m[:512] if len(m) > 512 else m for m in messages]

    print(f"Analyzing {len(messages)} commit messages...")

    # Process in batches
    batch_size = 32
    results = []

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            # If batch fails, process individually
            for msg in batch:
                try:
                    result = sentiment_pipeline(msg[:200])  # Truncate more
                    results.extend(result)
                except:
                    results.append({'label': 'NEUTRAL', 'score': 0.5})

        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i+batch_size, len(messages))}/{len(messages)}")

    # Analyze results
    sentiments = pd.DataFrame(results)

    positive = (sentiments['label'] == 'POSITIVE').sum()
    negative = (sentiments['label'] == 'NEGATIVE').sum()

    print(f"\nSentiment Distribution:")
    print(f"  - Positive: {positive} ({positive/len(sentiments)*100:.1f}%)")
    print(f"  - Negative: {negative} ({negative/len(sentiments)*100:.1f}%)")

    # Add sentiment to dataframe
    df_with_sentiment = df.copy()
    df_with_sentiment = df_with_sentiment.dropna(subset=['message']).head(len(results))
    df_with_sentiment['sentiment'] = [r['label'] for r in results[:len(df_with_sentiment)]]
    df_with_sentiment['sentiment_score'] = [r['score'] for r in results[:len(df_with_sentiment)]]

    # Analyze sentiment by commit type
    print("\nSentiment by Commit Type:")

    def get_commit_type(msg):
        if pd.isna(msg):
            return 'other'
        msg = msg.lower()
        for prefix in ['feat', 'fix', 'docs', 'refactor', 'test', 'chore', 'style', 'perf']:
            if msg.startswith(prefix):
                return prefix
        return 'other'

    df_with_sentiment['commit_type'] = df_with_sentiment['message'].apply(get_commit_type)

    sentiment_by_type = df_with_sentiment.groupby('commit_type').agg({
        'sentiment': lambda x: (x == 'POSITIVE').mean() * 100,
        'sentiment_score': 'mean'
    }).round(2)
    sentiment_by_type.columns = ['positive_pct', 'avg_confidence']
    print(sentiment_by_type.to_string())

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall sentiment
    labels = ['Positive', 'Negative']
    sizes = [positive, negative]
    colors = ['#66b3ff', '#ff9999']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0))
    axes[0].set_title('BERT Sentiment Analysis\nof Commit Messages')

    # Sentiment by commit type
    type_order = ['feat', 'fix', 'docs', 'refactor', 'test', 'chore', 'other']
    sentiment_by_type = sentiment_by_type.reindex([t for t in type_order if t in sentiment_by_type.index])

    colors = ['#66b3ff' if p > 50 else '#ff9999' for p in sentiment_by_type['positive_pct']]
    bars = axes[1].bar(sentiment_by_type.index, sentiment_by_type['positive_pct'],
                       color=colors, edgecolor='black')
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Commit Type')
    axes[1].set_ylabel('Positive Sentiment (%)')
    axes[1].set_title('Sentiment by Commit Type')
    axes[1].set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, sentiment_by_type['positive_pct']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bert_sentiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'bert_sentiment.png'}")

    # Find most negative commits (for qualitative analysis)
    print("\nMost Negative Commits (potential frustration indicators):")
    negative_commits = df_with_sentiment[df_with_sentiment['sentiment'] == 'NEGATIVE'].nlargest(5, 'sentiment_score')
    for _, row in negative_commits.iterrows():
        msg = row['message'][:80] + "..." if len(str(row['message'])) > 80 else row['message']
        print(f"  - {msg}")

    return {
        'positive_count': positive,
        'negative_count': negative,
        'positive_pct': positive/len(sentiments)*100,
        'negative_pct': negative/len(sentiments)*100,
        'sentiment_by_type': sentiment_by_type.to_dict()
    }


def find_failed_interpretations(df):
    """Look for patterns that DON'T exist (falsifiability)"""
    print("\n" + "="*60)
    print("FAILED INTERPRETATION ANALYSIS")
    print("="*60)

    # Hypothesis 1: Commit message length correlates with code churn
    # (We don't have churn data, so test with hour of day instead)

    df['msg_length'] = df['message'].fillna('').str.len()
    df['hour'] = pd.to_datetime(df['date']).dt.hour

    # Hypothesis: Longer messages at night (more thoughtful)
    night_commits = df[(df['hour'] >= 22) | (df['hour'] <= 6)]
    day_commits = df[(df['hour'] >= 9) & (df['hour'] <= 18)]

    night_avg = night_commits['msg_length'].mean()
    day_avg = day_commits['msg_length'].mean()

    print(f"\nHypothesis: Longer commit messages at night (more reflection)")
    print(f"  - Night (22:00-06:00) avg length: {night_avg:.1f} chars")
    print(f"  - Day (09:00-18:00) avg length: {day_avg:.1f} chars")
    print(f"  - Difference: {abs(night_avg - day_avg):.1f} chars ({((night_avg/day_avg)-1)*100:+.1f}%)")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(night_commits['msg_length'].dropna(),
                                       day_commits['msg_length'].dropna())
    print(f"  - t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

    if p_value > 0.05:
        print(f"  - RESULT: No significant difference (p > 0.05) - HYPOTHESIS REJECTED")
        failed_hypothesis_1 = True
    else:
        print(f"  - RESULT: Significant difference (p < 0.05)")
        failed_hypothesis_1 = False

    # Hypothesis 2: Weekend commits are more experimental (more 'feat')
    df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek >= 5

    def is_feat(msg):
        if pd.isna(msg):
            return False
        return msg.lower().startswith('feat')

    df['is_feat'] = df['message'].apply(is_feat)

    weekend_feat_rate = df[df['is_weekend']]['is_feat'].mean() * 100
    weekday_feat_rate = df[~df['is_weekend']]['is_feat'].mean() * 100

    print(f"\nHypothesis: Weekend commits are more experimental (higher feat ratio)")
    print(f"  - Weekend feat rate: {weekend_feat_rate:.1f}%")
    print(f"  - Weekday feat rate: {weekday_feat_rate:.1f}%")
    print(f"  - Difference: {weekend_feat_rate - weekday_feat_rate:+.1f}%")

    # Chi-square test
    contingency = pd.crosstab(df['is_weekend'], df['is_feat'])
    chi2, p_value2, dof, expected = stats.chi2_contingency(contingency)
    print(f"  - Chi-square: {chi2:.3f}, p-value: {p_value2:.3f}")

    if p_value2 > 0.05:
        print(f"  - RESULT: No significant difference (p > 0.05) - HYPOTHESIS REJECTED")
        failed_hypothesis_2 = True
    else:
        print(f"  - RESULT: Significant difference (p < 0.05)")
        failed_hypothesis_2 = False

    return {
        'night_vs_day_msg_length': {
            'night_avg': night_avg,
            'day_avg': day_avg,
            'p_value': p_value,
            'rejected': failed_hypothesis_1
        },
        'weekend_feat_rate': {
            'weekend': weekend_feat_rate,
            'weekday': weekday_feat_rate,
            'p_value': p_value2,
            'rejected': failed_hypothesis_2
        }
    }


def main():
    print("="*60)
    print("ENHANCED MSR ANALYSIS")
    print("="*60)

    # Load commits
    df = load_commits()
    print(f"Loaded {len(df)} commits")

    results = {}

    # 1. Refactor temporal patterns
    results['refactor_patterns'] = analyze_refactor_temporal_patterns(df)

    # 2. BERT sentiment analysis
    results['bert_sentiment'] = run_bert_sentiment_analysis(df)

    # 3. Failed interpretations
    results['failed_interpretations'] = find_failed_interpretations(df)

    # Save results
    with open(DATA_DIR / 'enhanced_analysis_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\n\nResults saved to: {DATA_DIR / 'enhanced_analysis_results.json'}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
