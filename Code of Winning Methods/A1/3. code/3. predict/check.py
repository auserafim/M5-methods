import numpy as np
import pandas as pd
import os

# =========================
# PATHS
# =========================
DATA_DIR = "/app/A1/2. data"

submission_path = os.path.join(DATA_DIR, "submission_final.csv")
sales_path = os.path.join(DATA_DIR, "sales_train_validation.csv")
calendar_path = os.path.join(DATA_DIR, "calendar.csv")
prices_path = os.path.join(DATA_DIR, "sell_prices.csv")

# =========================
# LOAD FILES
# =========================
y_pred = pd.read_csv(submission_path).set_index("id")
sales = pd.read_csv(sales_path)
calendar = pd.read_csv(calendar_path)
prices = pd.read_csv(prices_path)

# =========================
# TRUE VALUES (LAST 28 DAYS)
# =========================
d_cols = [c for c in sales.columns if c.startswith("d_")]
last_28_cols = d_cols[-28:]

y_true = sales.set_index("id")[last_28_cols]
y_true.columns = [f"F{i}" for i in range(1, 29)]

# =========================
# ALIGN IDS
# =========================
common_ids = y_true.index.intersection(y_pred.index)
y_true = y_true.loc[common_ids]
y_pred = y_pred.loc[common_ids]

# =========================
# RMSSE SCALE (DENOMINATOR)
# =========================
train_cols = d_cols[:-28]
train = sales.set_index("id")[train_cols]

diff = train.diff(axis=1).iloc[:, 1:]
scale = (diff ** 2).mean(axis=1)
scale = scale.loc[common_ids]

# =========================
# RMSSE NUMERATOR
# =========================
se = ((y_true - y_pred) ** 2).mean(axis=1)
rmsse = np.sqrt(se / scale)

# =========================
# WEIGHTS (REVENUE LAST 28 DAYS)
# =========================
melted = sales[["id", "item_id", "store_id"] + last_28_cols]
melted = melted.melt(id_vars=["id", "item_id", "store_id"], var_name="d", value_name="sales")

melted = melted.merge(calendar[["d", "wm_yr_wk"]], on="d", how="left")
melted = melted.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

melted["revenue"] = melted["sales"] * melted["sell_price"]

weights = melted.groupby("id")["revenue"].sum()
weights = weights.loc[common_ids]
weights = weights / weights.sum()

# =========================
# WRMSSE (FINAL SCORE)
# =========================
wrmsse = np.sum(weights * rmsse)

print("WRMSSE (Kaggle-style):", wrmsse)
