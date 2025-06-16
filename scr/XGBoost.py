import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
from multiprocessing import cpu_count
import warnings
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

def select_features_by_rfecv(X, y, model=None, min_features_to_select=1, cv_splits=3):
    # デフォルトはXGBoostで
    if model is None:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
    rfecv = RFECV(
        estimator=model,
        step=1,
        min_features_to_select=min_features_to_select,
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    selected = list(X.columns[rfecv.support_])
    print(f"🖊️ RFECV Selected: {selected}（最適特徴数: {len(selected)}）")
    return selected


def drop_constant_features(X, threshold=1e-10):
    drop_cols = [col for col in X.columns if X[col].std() < threshold]
    if drop_cols:
        print(f"⚠️ 定数特徴量を除去: {drop_cols}")
    return X.drop(columns=drop_cols, errors='ignore')

def select_features_by_shap(model, X_train, cumulative_cut=0.90):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': X_train.columns, 'mean_abs_shap': mean_shap})
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df['cumsum'] = importance_df['mean_abs_shap'].cumsum() / importance_df['mean_abs_shap'].sum()
    selected = importance_df[importance_df['cumsum'] <= cumulative_cut]['feature'].tolist()
    if not selected:
        selected = list(X_train.columns)
    print(f"🖊️ SHAP Top features: {selected}")
    return selected

def select_features_by_rfe(X_train, y_train, n_features=None):
    if n_features is None:
        n_features = max(1, int(len(X_train.columns)*0.6))
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(X_train, y_train)
    selected = list(X_train.columns[selector.support_])
    print(f"🖊️ RFE Selected: {selected}")
    return selected

def adjusted_r2(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def sequential_xgboost(train_df, test_df, target_col="incidence", periods_order=None, n_jobs=None):
    if periods_order is None:
        periods_order = train_df["period"].unique()

    train_all, test_all = train_df.copy(), test_df.copy()
    meta_cols = ["brand", "year", "period", "date"]
    train_results, test_results = [], []

    if n_jobs is None:
        n_jobs = cpu_count()

    for i, period in enumerate(periods_order):
        print("\n--------------------------------------------------")
        print(f"🔎 Period: {period}")

        # インデックスをしっかり連番に
        train_current = train_all[train_all["period"] == period].copy().reset_index(drop=True)
        test_current = test_all[test_all["period"] == period].copy().reset_index(drop=True)
        y_train = train_current[target_col].reset_index(drop=True)
        y_test = test_current[target_col].reset_index(drop=True)
        print(f"📊 Train Samples: {len(y_train)}, Test Samples: {len(y_test)}")

        # 逐次自己回帰の予測値を追加
        for j in range(i):
            prev_period = periods_order[j]
            pred_col = f"{prev_period}_pred"
            fill_vals = train_all[pred_col].mean() if pred_col in train_all.columns else 0.0
            # 全体に埋める
            train_all[pred_col] = train_all[pred_col].fillna(fill_vals)
            test_all[pred_col] = test_all[pred_col].fillna(fill_vals)
            train_current[pred_col] = train_current[pred_col].fillna(fill_vals)
            test_current[pred_col] = test_current[pred_col].fillna(fill_vals)

        exclude_cols = meta_cols + [target_col]
        X_train_raw = train_current.drop(columns=exclude_cols, errors="ignore").dropna(axis=1, how="all")

        # 欠損除去後、indexリセットしてX,yを揃える
        valid_mask = X_train_raw.notna().all(axis=1)
        X_train_clean = X_train_raw[valid_mask].reset_index(drop=True)
        y_train_clean = y_train[valid_mask].reset_index(drop=True)

        if len(X_train_clean) == 0 or X_train_clean.shape[1] == 0:
            print("❌ 有効サンプルゼロ or 特徴量消失 → 平均代入")
            y_train_pred = np.full_like(y_train, np.mean(y_train), dtype=np.float64)
            y_test_pred = np.full_like(y_test, np.mean(y_train), dtype=np.float64)
            # 全indexに「平均値」を格納
            train_all[f"{period}_pred"] = np.mean(y_train)
            test_all[f"{period}_pred"] = np.mean(y_train)
            rfe_selected = []
        else:
            X_train_clean = drop_constant_features(X_train_clean)

            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=n_jobs)
            model.fit(X_train_clean, y_train_clean)

            shap_selected = select_features_by_shap(model, X_train_clean)
            X_rfe = X_train_clean[shap_selected]
            # rfe_selected = select_features_by_rfe(X_rfe, y_train_clean, n_features=int(len(X_rfe.columns)*0.6))
            # rfe_selected = select_features_by_rfecv(X_rfe, y_train_clean, model=model, min_features_to_select=1, cv_splits=3)
    
            # 再学習
            # model.fit(X_rfe[rfe_selected], y_train_clean)
            model.fit(X_rfe, y_train_clean)
            rfe_selected = X_rfe.columns.tolist()  # SHAPで選ばれた特徴量をそのまま使う

            # [1] train_allに、**全indexで予測値**を伝播
            all_train_X = train_all.drop(columns=meta_cols+[target_col], errors="ignore")
            for feat in rfe_selected:
                fill_val = X_rfe[feat].mean()
                all_train_X[feat] = all_train_X[feat].fillna(fill_val)
            all_train_X = all_train_X.reindex(columns=rfe_selected)
            train_all[f"{period}_pred"] = model.predict(all_train_X)

            # [2] test_allに、**全indexで予測値**を伝播
            all_test_X = test_all.drop(columns=meta_cols+[target_col], errors="ignore")
            for feat in rfe_selected:
                fill_val = X_rfe[feat].mean()
                all_test_X[feat] = all_test_X[feat].fillna(fill_val)
            all_test_X = all_test_X.reindex(columns=rfe_selected)
            test_all[f"{period}_pred"] = model.predict(all_test_X)

            # このperiod分の train/test current を評価
            y_train_pred = train_all.loc[train_current.index, f"{period}_pred"].values
            y_test_pred = test_all.loc[test_current.index, f"{period}_pred"].values

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        # R2計算
        if len(y_train) > 1:
            train_r2 = r2_score(y_train, y_train_pred)
            n_features = X_train_raw.shape[1]
            train_r2_adj = adjusted_r2(y_train, y_train_pred, n_features)
        else:
            train_r2 = np.nan
            train_r2_adj = np.nan

        train_results.append({
            "period": period,
            "rmse": train_rmse,
            "r2": train_r2,
            "r2_adj": train_r2_adj,
            "features": list(X_train_raw.columns)
        })
        test_results.append({
            "period": period,
            "rmse": test_rmse
        })

        print(f"✅ Train RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}")

    return pd.DataFrame(train_results), pd.DataFrame(test_results), train_all, test_all
