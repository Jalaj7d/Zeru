import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(json_file):
    with open(json_file, 'r') as f:
        content = f.read().strip()
        if not content.startswith('['):
            content = '[' + content
        if not content.endswith(']'):
            content = content + ']'
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON Error:", e)
        return pd.DataFrame()
    records = []
    for tx in raw_data:
        try:
            action_data = tx.get("actionData", {})
            amount_raw = action_data.get("amount", "0")
            amount = float(amount_raw) / 1e6 if amount_raw.isdigit() else 0
            records.append({
                "wallet": tx.get("userWallet"),
                "action": tx.get("action", "").lower(),
                "amount": amount,
                "asset_symbol": action_data.get("assetSymbol", "").upper(),
                "price_usd": float(action_data.get("assetPriceUSD", 0)),
                "timestamp": tx.get("timestamp")
            })
        except Exception as e:
            print("Error parsing tx:", e)
            continue
    df = pd.DataFrame(records)
    print(f"Loaded transactions: {len(df)}")
    return df

def feature_engineering(df):
    df.columns = df.columns.str.lower()
    grouped = df.groupby("wallet")
    features = pd.DataFrame()
    features["deposit_count"] = grouped["action"].apply(lambda x: (x == "deposit").sum())
    features["borrow_count"] = grouped["action"].apply(lambda x: (x == "borrow").sum())
    features["repay_count"] = grouped["action"].apply(lambda x: (x == "repay").sum())
    features["liquidation_count"] = grouped["action"].apply(lambda x: (x == "liquidationcall").sum())
    features["redeem_count"] = grouped["action"].apply(lambda x: (x == "redeemunderlying").sum())
    features["avg_deposit_amount"] = grouped.apply(lambda x: x[x["action"] == "deposit"]["amount"].mean() or 0)
    features["avg_borrow_amount"] = grouped.apply(lambda x: x[x["action"] == "borrow"]["amount"].mean() or 0)
    features["repay_ratio"] = features["repay_count"] / (features["borrow_count"] + 1e-5)
    features["borrow_to_deposit_ratio"] = features["borrow_count"] / (features["deposit_count"] + 1e-5)
    return features.fillna(0)

def score_wallets(features):
    WEIGHTS = {
        "deposit_count": 0.15,
        "borrow_count": 0.10,
        "repay_count": 0.15,
        "liquidation_count": -0.25,
        "redeem_count": 0.10,
        "avg_deposit_amount": 0.15,
        "avg_borrow_amount": -0.05,
        "repay_ratio": 0.15,
        "borrow_to_deposit_ratio": -0.10
    }
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    weight_vec = np.array([WEIGHTS[col] for col in features.columns])
    raw_scores = np.dot(scaled, weight_vec)
    norm_scores = MinMaxScaler(feature_range=(0, 1000)).fit_transform(raw_scores.reshape(-1, 1)).flatten()
    return pd.DataFrame({"wallet": features.index, "credit_score": norm_scores.astype(int)})


def main():
    input_file = "user-wallet-transactions.json"
    output_file = "wallet_scores.csv"
    df = load_data(input_file)
    if df.empty:
        print("No data loaded. Please check your file.")
        return
    features = feature_engineering(df)
    scores = score_wallets(features)
    scores.to_csv(output_file, index=False)
    print(f"Credit scores saved to {output_file}")
    print(scores.head())
    bins = list(range(0, 1100, 100))  # [0, 100, 200, ..., 1000]
    labels = [f"{i}-{i+100}" for i in bins[:-1]]
    scores["score_range"] = pd.cut(scores["credit_score"], bins=bins, labels=labels, right=False)
    score_dist = scores["score_range"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    score_dist.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Credit Score Distribution")
    plt.xlabel("Score Range")
    plt.ylabel("Number of Wallets")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


Output:

Loaded transactions: 100000
/tmp/ipython-input-5-2493056174.py:49: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  features["avg_deposit_amount"] = grouped.apply(lambda x: x[x["action"] == "deposit"]["amount"].mean() or 0)
/tmp/ipython-input-5-2493056174.py:50: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  features["avg_borrow_amount"] = grouped.apply(lambda x: x[x["action"] == "borrow"]["amount"].mean() or 0)
Credit scores saved to wallet_scores.csv
                                       wallet  credit_score
0  0x00000000001accfa9cef68cf5371a23025b6d4b6           277
1  0x000000000051d07a4fb3bd10121a343d85818da6           277
2  0x000000000096026fb41fc39f9875d164bd82e2dc           277
3  0x0000000000e189dd664b9ab08a33c4839953852c           276
4  0x0000000002032370b971dabd36d72f3e5a7bf1ee           444

<img width="989" height="590" alt="graph" src="https://github.com/user-attachments/assets/5ca2beec-7a2b-4784-9376-dc207bea6337" />
