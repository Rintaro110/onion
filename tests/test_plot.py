import pandas as pd
import matplotlib.pyplot as plt
import os


def save_results_to_excel(models, predictors_list, filename="regression_results.xlsx", folder="outputs"):
    """
    回帰分析の結果をExcelファイルに出力する関数（単体/リスト対応、標準化回帰係数追加）。

    Args:
        models (list): statsmodelsのモデルオブジェクトのリスト
        predictors_list (list): 説明変数のリストのリスト
        filename (str): 出力ファイル名
        folder (str): 出力フォルダ名（デフォルト: "outputs"）

    Returns:
        None
    """
    print("---------------------------------------------------")
    print("回帰分析の結果をExcelファイルに保存します。")

    # 出力フォルダの作成
    os.makedirs(folder, exist_ok=True)
    
    if not isinstance(models, list):
        models, predictors_list = [models], [predictors_list]

    results = []
    for i, (model, predictors) in enumerate(zip(models, predictors_list)):
        # モデル名を作成（例: Best1, Best2, ...）
        model_name = f"Best{i+1}"
        model_header = pd.DataFrame([[model_name]], columns=[""])  # 1行だけのDataFrame
        
        # モデルのサマリーを取得
        summary_info_df = pd.DataFrame(model.summary2().tables[0])

        # 回帰係数テーブルを取得
        summary_df = pd.DataFrame(model.summary2().tables[1])

        # 標準化回帰係数（Standardized Beta）の計算    
        std_X = model.model.exog[:, 1:].std(axis=0)
        std_Y = model.model.endog.std()
        standardized_betas = model.params[1:] * (std_X / std_Y)

        # テーブルの整形
        summary_df.columns = ["Coef.", "Std Err", "t", "P>|t|", "[0.025", "0.975]"]
        summary_df.insert(0, "Variable", model.params.index)
        summary_df["Standardized Beta"] = pd.Series(standardized_betas, index=model.params.index[1:])
        empty_row = pd.DataFrame([[""] * len(summary_df.columns)], columns=summary_df.columns)

        # 各モデルのデータを整理
        model_result = pd.concat([model_header, summary_info_df, summary_df, empty_row])
        results.append(model_result)

    # すべての結果を結合
    results_df = pd.concat(results, ignore_index=True)

    # Excelに保存
    file_path = os.path.join(folder, filename)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Regression Results")

    print(f"Results saved to {file_path}")
    print("---------------------------------------------------")


def plot_regression(models, X_list, y, years=None, filename_prefix="regression_plot", folder="outputs", show_years=False):
    """
    回帰直線をプロットして画像ファイルとして保存する関数（単体/リスト対応）。
    
    Args:
        models (list): statsmodelsのモデルオブジェクトのリスト
        X_list (list): 説明変数のリストのリスト
        y (pd.Series): 目的変数
        years (pd.Series or list, optional): 各データポイントの年度（プロットに表示可能）
        filename_prefix (str): 画像ファイルのプレフィックス
        folder (str): 出力フォルダ名（デフォルト: "outputs"）
        show_years (bool): 各点に年度を表示するか（デフォルト: False）

    Returns:
        None
    """
    print("---------------------------------------------------")
    print("回帰直線をプロットして保存します。")

    # 出力フォルダの作成
    os.makedirs(folder, exist_ok=True)
    
    if not isinstance(models, list):
        models, X_list = [models], [X_list]

    for i, (model, X) in enumerate(zip(models, X_list)):
        predictions = model.predict(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.6, edgecolors='w', label="Fitted vs Actual")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Ideal Fit")
        
        # 年度をプロット上に表示するオプション
        if show_years and years is not None:
            for actual, predicted, year in zip(y, predictions, years):
                plt.annotate(str(year), (actual, predicted), fontsize=6, ha='right', alpha=0.7)

        # 軸とタイトルの設定
        plt.xlabel('Actual Values')
        plt.ylabel('Fitted Values')
        plt.title(f'Regression Results {i+1}')
        
        # R-squared と Adjusted R-squared を表示
        plt.text(0.1, 0.9, f'R-squared: {model.rsquared:.2f}', transform=plt.gca().transAxes)
        plt.text(0.1, 0.85, f'Adj R-squared: {model.rsquared_adj:.2f}', transform=plt.gca().transAxes)

        plt.legend()
        plt.grid(True)

        # 画像を保存
        filename = f"{folder}/{filename_prefix}_{i+1}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")

    print("---------------------------------------------------")
