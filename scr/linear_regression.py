import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from itertools import combinations
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm

def calculate_vif(X):
    if X.shape[1] < 2:
        return True, []
    vif_scores = []
    for i, column in enumerate(X.columns):
        X_excluded = sm.add_constant(X.iloc[:, np.arange(X.shape[1]) != i])
        try:
            r_squared_i = sm.OLS(X.iloc[:, i], X_excluded).fit().rsquared
            vif = np.inf if r_squared_i > 0.999 else 1. / (1. - r_squared_i)
        except Exception:
            vif = np.inf
        vif_scores.append((column, vif))
    high_vif = [col for col, vif in vif_scores if vif > 10]
    return len(high_vif) == 0, high_vif

def evaluate_combination(args):
    combo, df, y = args
    try:
        X = df[list(combo)].copy()
        valid_vif, _ = calculate_vif(X)
        if not valid_vif:
            return None
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        adj_r2 = model.rsquared_adj
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return {
            "r2_adj": adj_r2,
            "rmse": rmse,
            "features": combo,
            "model": model
        }
    except Exception:
        return None

def sequential_linear_regression(df_all, period_order, target_period, max_features=4, n_jobs=None):
    results = []
    df_work = df_all.copy()
    meta_cols = ['brand', 'year', 'period']

    assert target_period in period_order, f"{target_period} not in period_order"

    end_index = period_order.index(target_period)
    n_jobs = cpu_count() if n_jobs is None else min(cpu_count(), n_jobs)

    for i in range(0, end_index + 1):
        current_period = period_order[i]
        df_current = df_work[df_work["period"] == current_period].copy()

        if df_current.empty:
            print(f"⚠️ No data for {current_period}")
            continue
        empty_cols = df_current.columns[df_current.isna().all()].tolist()

        if empty_cols:
            print (f"  - {len(empty_cols)} columns removed")
        df_current = df_current.drop(columns=empty_cols)

        y = df_current["incidence"]
        exclude_cols = meta_cols + ["incidence"]
        candidate_features = [c for c in df_current.columns if c not in exclude_cols]

        # Add all previous *_pred columns as candidate features
        pred_feature_cols = [col for col in df_current.columns if col.endswith("_pred")]
        candidate_features += pred_feature_cols

        all_args = [(combo, df_current, y)
                    for k in range(1, max_features + 1)
                    for combo in combinations(candidate_features, k)]

        with Pool(n_jobs) as pool:
            results_all = list(tqdm(pool.imap(evaluate_combination, all_args), total=len(all_args), desc=f"⏳ {current_period}"))
        valid_results = [res for res in results_all if res is not None]

        if not valid_results:
            print(f"❌ {current_period}: No valid model found")
            continue

        best = max(valid_results, key=lambda x: x["r2_adj"])
        best_metrics = {
            "period": current_period,
            "r2_adj": best["r2_adj"],
            "rmse": best["rmse"],
            "features": best["features"]
        }

        results.append(best_metrics)

        print(f"✅ {current_period}: R2_adj={best_metrics['r2_adj']:.3f}, RMSE={best_metrics['rmse']:.3f}")
        print(f"   📌 選ばれた特徴量: {list(best_metrics['features'])}")

        if i + 1 < len(period_order):
            X_next = df_work[df_work["period"] == period_order[i + 1]]
            if not X_next.empty:
                try:
                    pred = best["model"].predict(sm.add_constant(X_next[list(best["features"])]))
                    df_work.loc[X_next.index, f"{current_period}_pred"] = pred
                except Exception as e:
                    print(f"❌ Prediction error at {current_period} -> {period_order[i + 1]}: {e}")

    return pd.DataFrame(results)

def save_results_to_excel(results_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_excel(output_path, index=False)
    print(f"📁 結果を保存しました: {output_path}")
