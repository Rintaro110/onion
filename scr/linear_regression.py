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


# VIF計算
def check_constant_features(X, threshold=1e-8):
    consts = [col for col in X.columns if X[col].std() < threshold]
    nans = [col for col in X.columns if X[col].isna().all()]
    return consts, nans

# --- VIF計算 ---
def calculate_vif(X):
    if X.shape[1] < 2:
        return True, []
    vif_scores = []
    for i, col in enumerate(X.columns):
        X_excl = sm.add_constant(X.iloc[:, np.arange(X.shape[1]) != i])
        try:
            r2_i = sm.OLS(X.iloc[:, i], X_excl).fit().rsquared
            vif = np.inf if r2_i > 0.999 else 1./(1. - r2_i)
        except Exception:
            vif = np.inf
        vif_scores.append((col, vif))
    high_vif = [col for col, vif in vif_scores if vif > 10]
    return len(high_vif) == 0, high_vif

# --- Exhaustive search 1組評価 ---
def evaluate_combination(args):
    combo, df, y = args
    try:
        X = df[list(combo)].copy()
        valid_vif, _ = calculate_vif(X)
        consts, nans = check_constant_features(X)
        if consts or nans or not valid_vif:
            return None
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        adj_r2 = model.rsquared_adj
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return {"r2_adj": adj_r2, "rmse": rmse, "features": combo, "model": model}
    except Exception:
        return None

def sequential_regression(
    train_df, test_df, target_col="incidence", periods_order=None,
    max_features=4, n_jobs=None
):
    if periods_order is None:
        periods_order = train_df["period"].unique()
    train_all, test_all = train_df.copy(), test_df.copy()
    meta_cols = ["index","brand", "year", "period", "date", "log_incidence", "incidence"]
    train_results, test_results = [], []

    if n_jobs is None: n_jobs = cpu_count()

    for i, period in enumerate(periods_order):
        print("\n--------------------------------------------------")
        print(f"🔎 Period: {period}")

        train_current = train_all[train_all["period"] == period].copy().reset_index()
        test_current = test_all[test_all["period"] == period].copy().reset_index()
        y_train, y_test = train_current[target_col].values, test_current[target_col].values
        print(f"📊 Train Samples: {len(y_train)}, Test Samples: {len(y_test)}")

        # 前periodの予測値伝搬
        for j in range(i):
            prev_period = periods_order[j]
            pred_col = f"{prev_period}_pred"
            mean_val = train_all[pred_col].mean() if pred_col in train_all.columns else 0.0
            train_all[pred_col] = train_all[pred_col].fillna(mean_val)
            test_all[pred_col] = test_all[pred_col].fillna(mean_val)
            if pred_col not in train_current.columns:
                train_current[pred_col] = train_all.loc[train_current["index"], pred_col].values
            if pred_col not in test_current.columns:
                test_current[pred_col] = test_all.loc[test_current["index"], pred_col].values

        exclude_cols = meta_cols + [target_col]
        X_train_raw = train_current.drop(columns=exclude_cols, errors="ignore").dropna(axis=1, how="all")
        X_test_raw = test_current.drop(columns=exclude_cols, errors="ignore").dropna(axis=1, how="all")
        for j in range(i):
            pred_col = f"{periods_order[j]}_pred"
            if pred_col not in X_train_raw.columns and pred_col in train_current.columns:
                X_train_raw[pred_col] = train_current[pred_col]
            if pred_col not in X_test_raw.columns and pred_col in test_current.columns:
                X_test_raw[pred_col] = test_current[pred_col]

        # 定数やNaN特徴量を記録
        consts, nans = check_constant_features(X_train_raw)
        if consts:
            print(f"⚠️ Constant features removed before combo search: {consts}")
        if nans:
            print(f"⚠️ All-NaN features removed before combo search: {nans}")
        # drop them from candidate features
        X_train_raw = X_train_raw.drop(columns=consts+nans, errors="ignore")
        candidate_features = list(X_train_raw.columns)
        print(f"   🔧 Candidate features: {candidate_features} (total {len(candidate_features)})")

        if X_train_raw.shape[1] == 0:
            print("⚠️ No usable features → Mean substitute")
            y_train_pred = np.full_like(y_train, np.mean(y_train), dtype=np.float64)
            y_test_pred = np.full_like(y_test, np.mean(y_train), dtype=np.float64)
            selected_features, r2_adj, rmse = [], np.nan, np.sqrt(np.mean((y_train - y_train_pred) ** 2))
            model = None
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
                print(f"   原因候補: {consts=}, {nans=}")
                y_train_pred = np.full_like(y_train, np.mean(y_train))
                y_test_pred = np.full_like(y_test, np.mean(y_train))
                selected_features, r2_adj, rmse = [], np.nan, np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                model = None
            else:
                best = max(valid_results, key=lambda x: x["r2_adj"])
                selected_features = list(best["features"])
                r2_adj, rmse = best["r2_adj"], best["rmse"]
                model = best["model"]

                # [★追加] 全行に予測値を埋める
                # --- train_all
                all_train_X = train_all.reindex(columns=selected_features).copy()
                for feat in selected_features:
                    # 埋まってない所をmeanで埋める（fitしたtrain_currentに準拠）
                    all_train_X[feat] = all_train_X[feat].fillna(train_current[feat].mean())
                all_train_X = sm.add_constant(all_train_X, has_constant='add')
                train_all[f"{period}_pred"] = model.predict(all_train_X)

                # --- test_all
                all_test_X = test_all.reindex(columns=selected_features).copy()
                for feat in selected_features:
                    all_test_X[feat] = all_test_X[feat].fillna(train_current[feat].mean())
                all_test_X = sm.add_constant(all_test_X, has_constant='add')
                test_all[f"{period}_pred"] = model.predict(all_test_X)

                # サブセットの評価値（現状のperiod用）
                X_train = sm.add_constant(train_current[selected_features], has_constant='add')
                y_train_pred = model.predict(X_train)

                X_test = test_current.reindex(columns=selected_features)
                for feat in selected_features:
                    X_test[feat] = X_test[feat].fillna(train_current[feat].mean())
                X_test = sm.add_constant(X_test, has_constant='add')
                y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        train_results.append({"period": period, "r2_adj": r2_adj, "rmse": train_rmse, "features": selected_features, "model": model})
        test_results.append({"period": period, "rmse": test_rmse})

        print(f"🖊️ Selected: {selected_features} | Train R2_adj={r2_adj:.3f} | Train RMSE={train_rmse:.3f} | Test RMSE={test_rmse:.3f}")

    return pd.DataFrame(train_results), pd.DataFrame(test_results), train_all, test_all