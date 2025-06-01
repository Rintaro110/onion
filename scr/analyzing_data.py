from itertools import combinations
import statsmodels.api as sm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def calculate_vif(X):
    """Check for multicollinearity using VIF."""
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

def sequential_linear_regression(df_all, period_order, target_period, max_features=4):
    """
    Perform stepwise sequential linear regression up to the target_period.

    Parameters
    ----------
    df_all : pd.DataFrame
        Preprocessed stepwise prediction dataset
    period_order : list of str
        Periods in chronological order
    target_period : str
        Final prediction target period
    max_features : int
        Max number of features to consider in each combination

    Returns
    -------
    pd.DataFrame
        Summary table with period, selected features, R²_adj, RMSE
    """
    results = []
    df_work = df_all.copy()
    meta_cols = ['brand', 'year', 'period']
    pred_col_name = f"{target_period}_incidence"

    assert target_period in period_order, f"{target_period} not in period_order"
    end_index = period_order.index(target_period)

    # For each step in period
    for i in range(0, end_index + 1):
        current_period = period_order[i]
        df_current = df_work[df_work["period"] == current_period].copy()
        if df_current.empty:
            print(f"⚠️ No data for {current_period}")
            continue

        # Remove empty columns
        empty_cols = df_current.columns[df_current.isna().all()].tolist()
        if empty_cols:
            # print(f"⚠️ {current_period}: 空の列を削除しました: {empty_cols}")
            print (f"  - {len(empty_cols)} columns removed")
        df_current = df_current.drop(columns=empty_cols)

        # Target and predictors
        y = df_current["incidence"]
        exclude_cols = meta_cols + ["incidence"]
        candidate_features = [c for c in df_current.columns if c not in exclude_cols]

        best_score = -np.inf
        best_model = None
        best_features = []
        best_metrics = {}

        pbar = tqdm(total=sum(1 for i in range(1, max_features+1) for _ in combinations(candidate_features, i)),
                    desc=f"⏳ {current_period}: Searching feature combinations")

        for k in range(1, max_features+1):
            for combo in combinations(candidate_features, k):
                X = df_current[list(combo)].copy()
                valid_vif, _ = calculate_vif(X)
                if not valid_vif:
                    pbar.update(1)
                    continue

                X = sm.add_constant(X)
                try:
                    model = sm.OLS(y, X).fit()
                    y_pred = model.predict(X)
                    adj_r2 = model.rsquared_adj
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    if adj_r2 > best_score:
                        best_score = adj_r2
                        best_model = model
                        best_features = list(combo)
                        best_metrics = {
                            "period": current_period,
                            "r2_adj": adj_r2,
                            "rmse": rmse,
                            "features": combo
                        }
                except Exception:
                    pass
                pbar.update(1)

        pbar.close()

        if best_model is None:
            print(f"❌ {current_period}: No valid model found")
            continue

        print(f"✅ {current_period}: R2_adj={best_metrics['r2_adj']:.3f}, RMSE={best_metrics['rmse']:.3f}")
        results.append(best_metrics)

        # Add prediction to next period
        X_next = df_work[df_work["period"] == period_order[i+1]] if i + 1 < len(period_order) else None
        if X_next is not None and not X_next.empty:
            pred = best_model.predict(sm.add_constant(X_next[best_features]))
            df_work.loc[X_next.index, f"{current_period}_pred"] = pred

    return pd.DataFrame(results)

