#!/usr/bin/env python3
"""
Recommendation Engine Demo Script

Usage:
    python demo_recommendations.py [profile_index]

Example:
    python demo_recommendations.py 0      # Get recommendations for first profile
    python demo_recommendations.py 42     # Get recommendations for profile at index 42
    python demo_recommendations.py        # Interactive mode
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.models.compatibility import recommend_profiles, validate_recommendations


def main():
    # Load data
    print("Loading data...")
    X = np.load('data/processed/X_processed.npy')
    labels = np.load('data/processed/cluster_labels.npy')
    df = pd.read_csv('data/raw/profiles.csv')
    
    print(f"Loaded {len(df)} profiles\n")
    
    # Get query index from command line or interactive
    if len(sys.argv) > 1:
        try:
            query_idx = int(sys.argv[1])
            if query_idx < 0 or query_idx >= len(df):
                print(f"Index out of range. Please choose 0-{len(df)-1}")
                return
        except ValueError:
            print("Invalid index. Please enter a number.")
            return
    else:
        # Interactive mode - show first 20 profiles
        print("Available profiles (first 20):")
        print("-" * 60)
        for i in range(20):
            p = df.iloc[i]
            print(f"  [{i:5d}] {p['name']:20s} | {p['current_role']:25s} | {p['industry']}")
        print("-" * 60)
        
        while True:
            try:
                query_idx = input("\nEnter profile index (or 'q' to quit): ").strip()
                if query_idx.lower() == 'q':
                    return
                query_idx = int(query_idx)
                if 0 <= query_idx < len(df):
                    break
                else:
                    print(f"Please enter 0-{len(df)-1}")
            except ValueError:
                print("Please enter a valid number.")
    
    # Get query profile
    query = df.iloc[query_idx]
    
    print("\n" + "=" * 70)
    print("QUERY PROFILE")
    print("=" * 70)
    print(f"  Index:       {query_idx}")
    print(f"  Name:        {query['name']}")
    print(f"  Role:        {query['current_role']}")
    print(f"  Company:     {query['current_company']}")
    print(f"  Industry:    {query['industry']}")
    print(f"  Seniority:   {query['seniority_level']}")
    print(f"  Experience:  {query['years_experience']} years")
    print(f"  Location:    {query['location']}")
    print(f"  Skills:      {query['skills']}")
    print(f"  Goals:       {query['goals']}")
    print(f"  Needs:       {query['needs']}")
    print(f"  Can Offer:   {query['can_offer']}")
    print(f"  Cluster:     {labels[query_idx]}")
    
    # Get recommendations
    print("\n" + "=" * 70)
    print("FINDING RECOMMENDATIONS...")
    print("=" * 70)
    
    recommendations = recommend_profiles(
        query_idx=query_idx,
        X=X,
        labels=labels,
        df_original=df,
        top_n=5,
        normalize_features=True  # This prevents feature block domination
    )
    
    print("\nTOP 5 RECOMMENDED PROFILES")
    print("-" * 70)
    
    for i, row in recommendations.iterrows():
        print(f"\n  #{i+1}: {row['name']}")
        print(f"      Similarity:  {row['similarity_score']:.4f}")
        print(f"      Role:       {row['current_role']} at {row['current_company']}")
        print(f"      Industry:   {row['industry']}")
        print(f"      Seniority:  {row['seniority_level']}")
        print(f"      Experience: {row['years_experience']} years")
        print(f"      Location:   {row['location']}")
        print(f"      Skills:     {row['skills']}")
    
    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    
    validation = validate_recommendations(query_idx, X, labels, df, recommendations)
    
    print(f"\nQuery Role:       {validation['query_role']}")
    print(f"Query Industry:   {validation['query_industry']}")
    print(f"Query Experience: {validation['query_experience']} years")
    print(f"Query Skills:     {validation['query_skills_count']}")
    
    print(f"\nRecommendation Quality Assessment:")
    print("-" * 70)
    
    all_industry_match = all(r['industry_match'] for r in validation['recommendations'])
    avg_skill_overlap = np.mean([r['skill_overlap'] for r in validation['recommendations']])
    avg_exp_diff = np.mean([r['experience_diff'] for r in validation['recommendations']])
    
    print(f"  Industry Match Rate: {'✓ 100%' if all_industry_match else '✗ Some mismatches'}")
    print(f"  Avg Skill Overlap:   {avg_skill_overlap:.1f} skills")
    print(f"  Avg Experience Diff:  {avg_exp_diff:.1f} years")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    Quality indicators:
    - High similarity score (>0.5) = good overall match
    - Same industry = relevant for professional networking
    - Skill overlap > 0 = shared technical interests
    - Experience diff < 3 years = comparable career stage
    
    Note: Role match may be different because clustering is based on
    overall profile similarity, not just job title.
    """)
    
    # Ask for another query
    print("\n" + "=" * 70)
    again = input("Get recommendations for another profile? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("Thank you!")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("\n\nDemo completed! Run with specific index: python demo_recommendations.py 42")
