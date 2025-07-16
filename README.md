Method Chosen

The system uses feature-based scoring with rule-weighted normalization:
1) Behavior-based Features: Extract user behavior like deposit_count, repay_ratio, and avg_borrow_amount.
2) Weighted Scoring: Apply manually defined weights to each feature.
3) Normalization: Scale scores across all users using MinMax scaling.
4) Final Output: Each wallet gets a score between 0 (risky) and 1000 (safe).

Architecture Overview

JSON File (Raw Transactions)

1) Load & Parse JSON
2) Feature Engineering
3) Weighted Scoring System
4) Normalize Scores (0–1000)
5) Output CSV (wallet_scores.csv)

Processing Flow

1. load_data()
Loads a JSON file of transactions.
Extracts:
  wallet address (userWallet)
  action (e.g., deposit, borrow)
  amount (scaled for 6 decimals)
  price in USD (for future optional features)
  Returns a DataFrame of structured transactions.

2. feature_engineering(df)
Groups data by wallet.
Computes behavioral features per wallet:
  Transaction counts for each action type.
  Average amounts.
  Ratios like repay/borrow.
  Returns a wallet-wise feature matrix.

3. score_wallets(features)
Applies predefined weights for each feature:
{
  "deposit_count": 0.15,
  "borrow_count": 0.10,
  "repay_count": 0.15,
  "liquidation_count": -0.25,
   and more...
}
Combines scaled feature values with weights.
Normalizes final scores to 0–1000 scale.
Outputs a score for each wallet.

4. main()
Calls all modules end-to-end:
  Loads data
  Extracts features
  Computes credit scores
  Saves results to wallet_scores.csv
