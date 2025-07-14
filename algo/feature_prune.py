def prune_lagged_features(X, importance_df, importance_threshold=0):
    important_feats = importance_df[importance_df.importance > importance_threshold]['decoded_name'].tolist()
    pruned = X.loc[:, X.columns.intersection(important_feats)]
    print(f"Kept {len(pruned.columns)} out of {X.shape[1]} features (threshold={importance_threshold})")
    return pruned

# Usage:
#X_pruned = prune_lagged_features(X, importance_df, importance_threshold=0)
