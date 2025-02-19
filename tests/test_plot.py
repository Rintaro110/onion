import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np


def plot_regression_results(X, y, model, filename="plot1.png", folder="../results"):
    """
    与えられたX, y, およびモデルを用いて回帰結果をプロットし、画像として保存する。
    
    :param X: 説明変数のデータフレーム
    :param y: 従属変数のデータフレーム
    :param model: statsmodelsの回帰モデルオブジェクト
    :param filename: 保存する画像ファイルの名前
    """

    # モデルに使用した説明変数を確認
    print("Model Variables (Training):", model.model.exog_names)

    # 予測時の説明変数を確認
    print("Prediction Variables:", X.columns)

    # サイズの一致を確認し、不一致の場合はインデックスで揃える
    if X.shape[0] != y.shape[0]:
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]


    # 定数項を追加して予測
    if 'const' not in model.model.exog_names:
        X_with_const = sm.add_constant(X, has_constant='add')
    else:
        X_with_const = X


    predictions = model.predict(X_with_const)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.6, edgecolors='w', label="Fitted vs Actual")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Ideal Fit")
    plt.xlabel('Actual Values')
    plt.ylabel('Fitted Values')
    plt.title('Regression Results')

    # Plot diagnostic lines or details
    plt.text(0.1, 0.9, f'R-squared: {model.rsquared:.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'Adj R-squared: {model.rsquared_adj:.2f}', transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/regression_{filename}")

    plt.close()
    print(f"Plot saved as regression_{filename}")

    # 残差プロット
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y - predictions, alpha=0.6, edgecolors='w', label="Residuals")
    plt.axhline(0, color='k', linestyle='--', lw=2)
    plt.xlabel('Measured')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/residuals_{filename}")
    plt.close()
    print(f"Plot saved as residuals_{filename}")

def plot_regression_result(X, y, model, filename="plot1.png", folder="../results"):
    """
    与えられたX, y, およびモデルを用いて回帰結果をプロットし、画像として保存する。
    
    :param X: 説明変数のデータフレーム
    :param y: 従属変数のデータフレームまたはシリーズ
    :param model: statsmodelsの回帰モデルオブジェクト
    :param filename: 保存する画像ファイルの名前
    :param folder: 保存先のフォルダ
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # モデルに使用した変数を取得
    model_exog_names = model.model.exog_names

    # 定数項の確認と追加
    if 'const' in model_exog_names:
        X_with_const = sm.add_constant(X, has_constant='add')
    else:
        X_with_const = sm.add_constant(X)

    # X_with_const の列名をモデルの変数名と一致させる
    X_with_const = X_with_const[model_exog_names]

    # 予測値の計算
    predictions = model.predict(X_with_const)

    # フォルダが存在しない場合、作成
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 実測値 vs 予測値のプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.6, edgecolors='w', label="Fitted vs Actual")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Ideal Fit")
    plt.xlabel('Actual Values')
    plt.ylabel('Fitted Values')
    plt.title('Regression Results')

    # プロットにR二乗値を追加
    plt.text(0.1, 0.9, f'R-squared: {model.rsquared:.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'Adj R-squared: {model.rsquared_adj:.2f}', transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/regression_{filename}")
    plt.close()
    print(f"Plot saved as {folder}/regression_{filename}")

    # 残差プロット
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y - predictions, alpha=0.6, edgecolors='w', label="Residuals")
    plt.axhline(0, color='k', linestyle='--', lw=2)
    plt.xlabel('Measured')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/residuals_{filename}")
    plt.close()
    print(f"Plot saved as {folder}/residuals_{filename}")

def plot_top_predictors(df, model, predictors):
    """
    最も影響力の強い説明変数を用いて目的変数と回帰直線をプロット。

    Args:
        df (pd.DataFrame): データフレーム。
        model (sm.OLS): 回帰モデル。
        predictors (list): 選択された説明変数。

    Returns:
        None
    """
    # モデルの影響度（係数の絶対値）を取得
    coefs = pd.Series(model.params[1:], index=model.model.exog_names[1:])
    top_predictors = coefs.abs().sort_values(ascending=False).index[:2]

    for predictor in top_predictors:
        plt.figure()
        
        # 横軸となる説明変数
        X = df[[predictor]]
        y = df['発病率']

        # モデルで学習に使用した全ての説明変数を基に予測
        X_with_const = sm.add_constant(df[model.model.exog_names[1:]])
        y_pred = model.predict(X_with_const)

        # 実際のデータと回帰直線のプロット
        plt.scatter(df[predictor], y, alpha=0.7, label='Actual Data')
        plt.plot(df[predictor], y_pred[:len(df)], color='red', label='Regression Line')  # 対応するデータ範囲のみプロット
        plt.title(f"発病率 vs {predictor}")
        plt.xlabel(predictor)
        plt.ylabel('発病率')
        plt.legend()
        plt.show()

def plot_top_predictors_with_fixed_regression(df, model, predictors):
    """
    回帰直線を正しく描画するため、データを昇順にソートして回帰直線を手動でプロット。

    Args:
        df (pd.DataFrame): データフレーム。
        model (sm.OLS): 回帰モデル。
        predictors (list): 選択された説明変数。

    Returns:
        None
    """

    # モデルの影響度（係数の絶対値）を取得
    coefs = pd.Series(model.params[1:], index=model.model.exog_names[1:])
    top_predictors = coefs.abs().sort_values(ascending=False).index[:2]

    for predictor in top_predictors:
        # データをX軸でソート
        sorted_df = df.sort_values(by=predictor)
        X = sorted_df[model.model.exog_names[1:]]  # モデルが使用した全ての説明変数
        y = sorted_df['発病率']

        # 定数項を追加
        X_with_const = sm.add_constant(X)

        # 回帰直線の予測値を計算
        y_pred = model.predict(X_with_const)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.scatter(sorted_df[predictor], y, alpha=0.7, label="Actual Data", edgecolor="k")
        plt.plot(sorted_df[predictor], y_pred, color="red", label="Regression Line")
        plt.title(f"発病率 vs {predictor}", fontsize=14)
        plt.xlabel(predictor, fontsize=12)
        plt.ylabel("発病率", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

def plot_standardized_coefficients(model, X, Y):
    """
    標準化された回帰係数を棒グラフでプロットし、数値も表示。

    :param model: statsmodels の回帰モデル
    :param X: 説明変数の DataFrame
    :param Y: 目的変数の Series
    """
    # **モデルで使用した説明変数のみを取得**
    explanatory_vars = model.model.exog_names[1:]  # 0番目は 'const' なので除外
    X = X[explanatory_vars]  # `X` をモデルの変数に揃える

    coefs = model.params[1:]  # 定数項を除外
    X_std = np.std(X, axis=0, ddof=1)  # 説明変数の標準偏差
    Y_std = np.std(Y, ddof=1)  # 目的変数の標準偏差

    # **サイズチェック**
    if len(coefs) != len(X_std):
        raise ValueError(f"Mismatch: coefs ({len(coefs)}) vs X_std ({len(X_std)})")

    # **標準化回帰係数の計算**
    standardized_coefs = coefs * (X_std / Y_std)

    # **NaN を除外**
    mask = ~np.isnan(standardized_coefs)
    standardized_coefs = standardized_coefs[mask]
    explanatory_vars = np.array(X.columns)[mask]

    if len(standardized_coefs) == 0:
        print("No valid coefficients to plot.")
        return

    # **標準化回帰係数をコンソールに表示**
    print("標準化回帰係数:")
    for var, coef in zip(explanatory_vars, standardized_coefs):
        print(f"{var}: {coef:.4f}")

    # **棒グラフを描画**
    plt.figure(figsize=(12, 6))
    bars = plt.barh(explanatory_vars, standardized_coefs, color='skyblue')
    plt.xlabel("Standardized Coefficient")
    plt.title("Feature Importance (Standardized)")
    plt.grid(True)

    # **各棒の横に数値を表示**
    for bar, coef in zip(bars, standardized_coefs):
        plt.text(bar.get_width() + 0.02 * np.sign(bar.get_width()),  # 棒の長さに応じて位置調整
                 bar.get_y() + bar.get_height()/2, 
                 f"{coef:.4f}", 
                 va='center', ha='left' if coef >= 0 else 'right')

    plt.show()

def plot_explanatory_vs_target(df, model, target_var):
    """
    説明変数 vs 目的変数を1つずつプロットし、回帰直線を描画。

    :param df: データフレーム
    :param model: statsmodelsの回帰モデル
    :param target_var: 目的変数の名前
    """
    explanatory_vars = model.model.exog_names[1:]  # 定数項を除く説明変数
    for var in explanatory_vars:
        plt.figure(figsize=(8, 6))
        sns.regplot(x=df[var], y=df[target_var], ci=None, line_kws={"color": "red"})
        plt.title(f"{target_var} vs {var}")
        plt.xlabel(var)
        plt.ylabel(target_var)
        plt.grid(True)
        plt.show()
