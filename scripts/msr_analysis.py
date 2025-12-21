#!/usr/bin/env python3
"""
MSR Analysis Script for Developer Mental Model Framework Study
Generates quantitative analyses of GitHub repository data.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

# Configuration
GITHUB_USER = "anderson-ufrj"
OUTPUT_DIR = "../figures"
DATA_DIR = "../data"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_repos(username):
    """Fetch all public repositories for a user."""
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    return repos

def fetch_commits(username, repo_name, max_commits=500):
    """Fetch commits for a repository."""
    commits = []
    page = 1
    while len(commits) < max_commits:
        url = f"https://api.github.com/repos/{username}/{repo_name}/commits?per_page=100&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        commits.extend(data)
        page += 1
    return commits[:max_commits]

def analyze_commit_patterns(commits_df):
    """Analyze temporal patterns in commits."""
    if commits_df.empty:
        return None

    commits_df['datetime'] = pd.to_datetime(commits_df['date'])
    commits_df['hour'] = commits_df['datetime'].dt.hour
    commits_df['dayofweek'] = commits_df['datetime'].dt.dayofweek
    commits_df['month'] = commits_df['datetime'].dt.to_period('M')

    return commits_df

def create_commit_heatmap(commits_df):
    """Create a heatmap of commit activity by day of week and hour."""
    if commits_df.empty:
        return

    # Create pivot table
    heatmap_data = commits_df.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)

    # Ensure all hours are present
    for h in range(24):
        if h not in heatmap_data.columns:
            heatmap_data[h] = 0
    heatmap_data = heatmap_data.reindex(columns=range(24))

    # Ensure all days are present
    for d in range(7):
        if d not in heatmap_data.index:
            heatmap_data.loc[d] = 0
    heatmap_data = heatmap_data.sort_index()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data,
                cmap='YlOrRd',
                annot=False,
                xticklabels=[f'{h:02d}' for h in range(24)],
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                ax=ax)
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_ylabel('Day of Week')
    ax.set_title('Commit Activity Heatmap (N={})'.format(len(commits_df)))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/commit_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/commit_heatmap.png")

def create_commits_over_time(commits_df):
    """Create a line chart of commits over time."""
    if commits_df.empty:
        return

    monthly = commits_df.groupby('month').size()

    fig, ax = plt.subplots(figsize=(12, 5))
    monthly.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Commits')
    ax.set_title('Commit Frequency Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/commits_over_time.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/commits_over_time.png")

def analyze_commit_messages(commits_df):
    """Analyze commit message patterns."""
    if commits_df.empty:
        return {}

    # Categorize by conventional commit types
    type_patterns = {
        'feat': r'^feat[:\(]',
        'fix': r'^fix[:\(]',
        'docs': r'^docs[:\(]',
        'refactor': r'^refactor[:\(]',
        'test': r'^test[:\(]',
        'chore': r'^chore[:\(]',
        'style': r'^style[:\(]',
        'perf': r'^perf[:\(]',
    }

    type_counts = defaultdict(int)
    for msg in commits_df['message']:
        msg_lower = msg.lower() if isinstance(msg, str) else ''
        categorized = False
        for type_name, pattern in type_patterns.items():
            import re
            if re.match(pattern, msg_lower):
                type_counts[type_name] += 1
                categorized = True
                break
        if not categorized:
            type_counts['other'] += 1

    return dict(type_counts)

def create_commit_type_chart(type_counts):
    """Create a pie chart of commit types."""
    if not type_counts:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by count
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_types]
    sizes = [t[1] for t in sorted_types]

    colors = plt.cm.Set3(range(len(labels)))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Commit Types Distribution (Conventional Commits)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/commit_types.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/commit_types.png")

def create_language_distribution(repos):
    """Create a chart of programming languages used."""
    lang_counts = defaultdict(int)
    for repo in repos:
        if repo.get('language'):
            lang_counts[repo['language']] += 1

    if not lang_counts:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    langs = [l[0] for l in sorted_langs]
    counts = [l[1] for l in sorted_langs]

    bars = ax.barh(langs[::-1], counts[::-1], color='steelblue', edgecolor='black')
    ax.set_xlabel('Number of Repositories')
    ax.set_ylabel('Language')
    ax.set_title('Top 10 Programming Languages by Repository Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/language_distribution.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/language_distribution.png")

def calculate_message_length_stats(commits_df):
    """Calculate commit message length statistics."""
    if commits_df.empty:
        return {}

    lengths = commits_df['message'].apply(lambda x: len(str(x)) if x else 0)

    return {
        'mean_length': lengths.mean(),
        'median_length': lengths.median(),
        'max_length': lengths.max(),
        'min_length': lengths.min(),
        'std_length': lengths.std()
    }

def create_message_length_histogram(commits_df):
    """Create histogram of commit message lengths."""
    if commits_df.empty:
        return

    lengths = commits_df['message'].apply(lambda x: len(str(x)) if x else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.1f}')
    ax.axvline(lengths.median(), color='green', linestyle='--', label=f'Median: {lengths.median():.1f}')
    ax.set_xlabel('Message Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Commit Message Length Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/message_length_histogram.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/message_length_histogram.png")

def generate_summary_stats(repos, commits_df):
    """Generate summary statistics."""
    stats = {
        'total_repos': len(repos),
        'total_commits': len(commits_df),
        'repos_with_commits': commits_df['repo'].nunique() if 'repo' in commits_df.columns else 0,
        'date_range': {
            'start': commits_df['date'].min() if not commits_df.empty else None,
            'end': commits_df['date'].max() if not commits_df.empty else None
        },
        'languages': {repo['language']: 1 for repo in repos if repo.get('language')},
        'avg_commits_per_repo': len(commits_df) / len(repos) if repos else 0
    }
    return stats

def main():
    print(f"Fetching repositories for {GITHUB_USER}...")
    repos = fetch_repos(GITHUB_USER)
    print(f"Found {len(repos)} repositories")

    # Save repos data
    with open(f'{DATA_DIR}/repos.json', 'w') as f:
        json.dump(repos, f, indent=2, default=str)

    # Fetch commits from all repos
    all_commits = []
    print("\nFetching commits from repositories...")
    for i, repo in enumerate(repos):
        repo_name = repo['name']
        print(f"  [{i+1}/{len(repos)}] {repo_name}...", end=' ')
        commits = fetch_commits(GITHUB_USER, repo_name, max_commits=200)
        print(f"{len(commits)} commits")
        for commit in commits:
            if commit.get('commit'):
                all_commits.append({
                    'repo': repo_name,
                    'sha': commit.get('sha', ''),
                    'message': commit['commit'].get('message', ''),
                    'date': commit['commit'].get('author', {}).get('date', ''),
                    'author': commit['commit'].get('author', {}).get('name', '')
                })

    # Create DataFrame
    commits_df = pd.DataFrame(all_commits)
    print(f"\nTotal commits collected: {len(commits_df)}")

    # Save commits data
    commits_df.to_csv(f'{DATA_DIR}/commits.csv', index=False)
    print(f"Saved: {DATA_DIR}/commits.csv")

    # Analyze patterns
    if not commits_df.empty:
        commits_df = analyze_commit_patterns(commits_df)

        # Generate visualizations
        print("\nGenerating visualizations...")
        create_commit_heatmap(commits_df)
        create_commits_over_time(commits_df)
        create_language_distribution(repos)
        create_message_length_histogram(commits_df)

        # Analyze commit types
        type_counts = analyze_commit_messages(commits_df)
        create_commit_type_chart(type_counts)

        # Message length stats
        msg_stats = calculate_message_length_stats(commits_df)

        # Summary stats
        summary = generate_summary_stats(repos, commits_df)
        summary['commit_types'] = type_counts
        summary['message_stats'] = msg_stats

        with open(f'{DATA_DIR}/summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved: {DATA_DIR}/summary_stats.json")

        # Print summary
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total repositories: {summary['total_repos']}")
        print(f"Total commits: {summary['total_commits']}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Avg commits/repo: {summary['avg_commits_per_repo']:.1f}")
        print(f"\nCommit types: {type_counts}")
        print(f"\nMessage length stats: {msg_stats}")
    else:
        print("No commits found!")

if __name__ == "__main__":
    main()
