import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool, cpu_count
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# 定数特徴量の検出
def check_constant_features(X, threshold=1e-8):
    return [col for col in X.columns if X[col].std() < threshold]

# VIF計算
def calculate_vif(X):
    if X.shape[1] < 2: return True, []
    vif_scores = []
    for i, col in enumerate(X.columns):
        X_excl = sm.add_constant(X.iloc[:, np.arange(X.shape[1]) != i])
        try:
            r2_i = sm.OLS(X.iloc[:, i], X_excl).fit().rsquared
            vif = np.inf if r2_i > 0.999 else 1./(1. - r2_i)
        except Exception: vif = np.inf
        vif_scores.append((col, vif))
    high_vif = [col for col, vif in vif_scores if vif > 10]
    return len(high_vif) == 0, high_vif

# Exhaustive search 1組評価
def evaluate_combination(args):
    combo, df, y = args
    try:
        X = df[list(combo)].copy()
        valid_vif, _ = calculate_vif(X)
        constant_cols = check_constant_features(X)
        if constant_cols or not valid_vif: return None
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        adj_r2 = model.rsquared_adj
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        return {"r2_adj": adj_r2, "rmse": rmse, "features": combo, "model": model}
    except Exception: return None

# 逐次自己回帰型 Exhaustive モデル本体
def sequential_regression(train_df, test_df, target_col="incidence", periods_order=None, max_features=4, n_jobs=None):
    if periods_order is None:
        periods_order = train_df["period"].unique()
    train_all, test_all = train_df.copy(), test_df.copy()
    meta_cols = ["brand", "year", "period", "date"]
    train_results, test_results = [], []

    if n_jobs is None: n_jobs = cpu_count()

    for i, period in enumerate(periods_order):
        print("\n--------------------------------------------------")
        print(f"🔎 Period: {period}")

        train_current = train_all[train_all["period"] == period].copy()
        test_current = test_all[test_all["period"] == period].copy()
        y_train, y_test = train_current[target_col].values, test_current[target_col].values
        print(f"📊 Train Samples: {len(y_train)}, Test Samples: {len(y_test)}")

        extra_cols = []
        for j in range(i):
            prev_period = periods_order[j]
            pred_col = f"{prev_period}_pred"
            if pred_col in train_all.columns:
                train_current[pred_col] = train_all.loc[train_current.index, pred_col].fillna(train_all[pred_col].mean())
            if pred_col in test_all.columns:
                test_current[pred_col] = test_all.loc[test_current.index, pred_col].fillna(test_all[pred_col].mean())
            extra_cols.append(pred_col)

        exclude_cols = meta_cols + [target_col]
        X_train_raw = train_current.drop(columns=exclude_cols, errors="ignore").dropna(axis=1, how="all")
        for extra in extra_cols:
            if extra not in X_train_raw.columns:
                X_train_raw[extra] = train_current[extra]

        candidate_features = list(X_train_raw.columns)
        print(f"   🔧 Candidate features: {candidate_features} (total {len(candidate_features)})")

        if X_train_raw.shape[1] == 0:
            print("⚠️ No usable features → Mean substitute")
            y_train_pred = np.full_like(y_train, np.mean(y_train), dtype=np.float64)
            y_test_pred = np.full_like(y_test, np.mean(y_train), dtype=np.float64)
        else:
            all_args = [(combo, train_current[candidate_features], y_train)
                        for k in range(1, max_features+1)
                        for combo in combinations(candidate_features, k)]
            print(f"   🔄 Exhaustive total combinations: {len(all_args)}")

            with Pool(n_jobs) as pool:
                results_all = list(tqdm(pool.imap(evaluate_combination, all_args),
                                         total=len(all_args), desc=f"⏳ {period}"))
            valid_results = [res for res in results_all if res is not None]

            if not valid_results:
                print("⚠️ No valid models → mean substitute")
                y_train_pred = np.full_like(y_train, np.mean(y_train))
                y_test_pred = np.full_like(y_test, np.mean(y_train))
                selected_features, r2_adj, rmse = [], 0.0, np.sqrt(np.mean((y_train - y_train_pred) ** 2))
            else:
                best = max(valid_results, key=lambda x: x["r2_adj"])
                selected_features = list(best["features"])
                r2_adj, rmse = best["r2_adj"], best["rmse"]
                model = best["model"]

                X_train = sm.add_constant(train_current[selected_features], has_constant='add')
                y_train_pred = model.predict(X_train)

                # test欠損埋め
                X_test_raw = test_current.reindex(columns=selected_features)
                for feat in selected_features:
                    X_test_raw[feat] = X_test_raw[feat].fillna(train_current[feat].mean())
                X_test = sm.add_constant(X_test_raw, has_constant='add')
                y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        train_results.append({"period": period, "r2_adj": r2_adj, "rmse": train_rmse, "features": selected_features})
        test_results.append({"period": period, "rmse": test_rmse})
        train_all.loc[train_current.index, f"{period}_pred"] = y_train_pred
        test_all.loc[test_current.index, f"{period}_pred"] = y_test_pred
        print(f"🖊️ Selected: {selected_features} | Train R2={r2_adj:.3f} | Train RMSE={train_rmse:.3f} | Test RMSE={test_rmse:.3f}")

    return pd.DataFrame(train_results), pd.DataFrame(test_results), train_all, test_all


