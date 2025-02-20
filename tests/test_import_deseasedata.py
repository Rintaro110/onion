import pandas as pd
from collections import defaultdict

def import_desease_data(file_path, start_year=None, end_year=None, target_names=["ターザン", "もみじ3号", "もみじの輝"], verbose=False):
    # Excelファイルを読み込む
    df = pd.read_excel(file_path)
    if verbose:
        print(f"データ読み込み完了: {file_path}")
        print(f"品種: {target_names}")

    # エラーハンドリング
    # 必要なカラムチェック
    if "年度" not in df.columns:
        raise ValueError("エクセルファイルに '年度' 列が見つかりません。列名を確認してください。")
    # ターゲット品種の存在チェック
    missing_columns = [col for col in target_names if col not in df.columns]
    if missing_columns:
        print(f"Warning: 指定された品種 {missing_columns} がエクセルファイルに存在しません。")
        target_names = [col for col in target_names if col in df.columns]
    # 年度範囲のバリデーション
    if start_year and end_year and start_year > end_year:
        raise ValueError(f"開始年度 ({start_year}) は終了年度 ({end_year}) よりも小さい必要があります。")

    # データを格納する辞書
    organized_data = defaultdict(lambda: defaultdict(list))
    years_collected = set()

    for _, row in df.iterrows():
        year = int(row["年度"])

        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue

        for variety in target_names:
            value = row[variety]
            if pd.isna(value):
                if verbose:
                    print(f"Warning: NaN value found in {variety} at year {year}")
                continue

            organized_data[variety][year].append(value)
            years_collected.add(year)

    # 取得されたデータの年度範囲
    if years_collected:
        min_year = min(years_collected)
        max_year = max(years_collected)
        print(f"取得されたデータの年度範囲: {min_year}年 ～ {max_year}年")
    else:
        print("Warning: 指定された範囲に有効なデータがありませんでした。")

    print("病害データ取得完了。")

    return organized_data




def calculate_correlation_and_plot(desease_data1, desease_data2):
    """
    2つの病害データセットから、同一の年度のデータの相関係数を計算し、
    散布図を生成して、各点に年度ラベルを付ける。
    
    :param desease_data1: 病害データセット1 (辞書形式)
    :param desease_data2: 病害データセット2 (辞書形式)
    """
    # 共通の品種と年度を取得
    for variety in desease_data1.keys() & desease_data2.keys():  # 共通の品種
        print(f"品種: {variety}")
        
        # 共通年度のデータを取得
        common_years = desease_data1[variety].keys() & desease_data2[variety].keys()  # 共通年度

        # データをリストに格納
        data1 = []
        data2 = []
        years = []

        for year in common_years:
            locations_data1 = desease_data1[variety][year]
            locations_data2 = desease_data2[variety][year]

            # 発病率の平均データを取得（各年度ごと）
            avg_data1 = next(data for location, data in locations_data1 if location == '平均')
            avg_data2 = next(data for location, data in locations_data2 if location == '平均')

            data1.append(avg_data1)
            data2.append(avg_data2)
            years.append(year)  # 年度を保持

        # データフレームを作成して相関係数を計算
        df = pd.DataFrame({'Data1': data1, 'Data2': data2}, index=years)
        correlation = df['Data1'].corr(df['Data2'])
        print(f"相関係数: {correlation:.2f}")

        # 散布図の作成
        plt.figure(figsize=(8, 6))
        plt.scatter(data1, data2, c='blue', alpha=0.6)

        # y=xの線を描く
        min_val = min(min(data1), min(data2))  # データの最小値
        max_val = max(max(data1), max(data2))  # データの最大値
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')  # y=xの線


        # 各点に年度を表示
        for i, year in enumerate(years):
            plt.text(data1[i], data2[i], str(year), fontsize=9, ha='right')

        plt.title(f"Correlation: {correlation:.2f}")
        plt.xlabel('desease data 1')
        plt.ylabel('desease data 2')
        plt.grid(True)
        plt.legend()  # 凡例を表示
        plt.show()

def plot_disease_rate_histogram(data, variety_name, use_average=True, bins=10):
    """
    全年度の病害率データを対象としたヒストグラムを作成する関数。
    
    Parameters:
        data (dict): 病害率データの辞書形式（品種ごとに年度・場所・病害率のタプルがリストとして格納）。
        variety_name (str): ヒストグラムを作成する品種の名前。
        use_average (bool): Trueの場合は、平均値を含むデータのみを使用する。
        bins (int): ヒストグラムのビンの数。
        
    Returns:
        None
    """
    # 指定された品種がデータに存在するか確認
    if variety_name not in data:
        print(f"データが見つかりません（品種: {variety_name}）")
        return
    
    # 全年度のデータを対象に病害率を抽出
    values = []
    for year, records in data[variety_name].items():
        if use_average:
            # 平均値のみを対象
            values.extend([rate for loc, rate in records if loc == '平均'])
        else:
            # 非平均値を対象
            values.extend([rate for loc, rate in records if loc != '平均'])
    
    # データがない場合はエラーメッセージを表示
    if not values:
        print(f"指定された条件に合致するデータがありません（品種: {variety_name}, 平均値使用: {use_average}）")
        return
    
    # ヒストグラムの作成
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(f"{variety_name} - 全年度の病害率ヒストグラム (平均値使用: {use_average})")
    plt.xlabel("病害率")
    plt.ylabel("頻度")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def calculate_outliers(data, use_average=True, method="IQR", theresh=3 ,n_neighbors=20):
    """
    全年度の病害率データを対象とした外れ値を計算する関数。
    
    Parameters:
        data (dict): 病害率データの辞書形式（品種ごとに年度・場所・病害率のタプルがリストとして格納）。
        use_average (bool): Trueの場合は、平均値を持つデータのみを使用する。
        method (str): 外れ値を計算する方法 ("IQR", "Z-score", "LOF")。
        n_neighbors (int): LOF法で使用する近傍点の数。
        
    Returns:
        dict: 外れ値が検出された品種、年度、場所およびその値を返す。
    """
    outliers = defaultdict(list)
    
    for variety, years_data in data.items():
        # 全ての年度の病害率データを統合
        values = []
        records = []
        for year, year_records in years_data.items():
            for loc, rate in year_records:
                if use_average and loc == '平均':
                    values.append(rate)
                    records.append((year, loc, rate))
                elif not use_average and loc != '平均':
                    values.append(rate)
                    records.append((year, loc, rate))
        
        # データがない場合はスキップ
        if not values:
            continue
        
        # 外れ値の検出
        if method == "IQR":
            # 四分位範囲 (IQR) 法
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            for year, loc, rate in records:
                if rate < lower_bound or rate > upper_bound:
                    outliers[variety].append((year, loc, rate))
            
            print(f"品種: {variety}, IQR法: 下限={lower_bound:.2f}, 上限={upper_bound:.2f}")
                    
        elif method == "Z-score":
            # 標準偏差 (Zスコア) 法
            mean = np.mean(values)
            std = np.std(values)
            for year, loc, rate in records:
                z_score = (rate - mean) / std
                if z_score > theresh:  # Zスコアが3以上のものを外れ値とする
                    outliers[variety].append((year, loc, rate))

            print(f"品種: {variety}, Zスコア法: 平均={mean:.2f}, 標準偏差={std:.2f}")
                    
        elif method == "LOF":
            # LOF (Local Outlier Factor) 法
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            values_array = np.array(values).reshape(-1, 1)
            lof_scores = lof.fit_predict(values_array)
            
            for i, (year, loc, rate) in enumerate(records):
                if lof_scores[i] == -1:  # 外れ値として検出されたデータ点
                    outliers[variety].append((year, loc, rate))
                    
        else:
            raise ValueError("Unsupported method. Choose 'IQR', 'Z-score', or 'LOF'.")
    
    return outliers


if __name__ == '__main__':

    # 使用例
    syukaku_data_path = 'resources/desease_data/disease_data_syukaku.xlsx'
    tyozou_data_path = "resources/desease_data/disease_data_tyozou.xlsx"
    start_year = 1990
    end_year = 2023
    target_varieties = ['ターザン']  # 取得したい品種を指定
    # desease_data1 = extract_turidama_disease_rate(file_path, start_year, end_year, target_varieties)
    # desease_data2 = extract_syuryou_desease_rate(directory_path, target_names=target_varieties)

    # Example usage
    data = import_desease_data(syukaku_data_path, start_year=start_year, end_year=end_year)
    # data2 = import_desease_data(tyozou_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties)

    # Print the output
    for variety, years_data in data.items():
        print(f"\n品種名: {variety}")
        for year, locations_data in years_data.items():
            print(f"  年度: {year}")
            for location, value in locations_data:
                if location == "avg":
                    print(f"    平均値: {value}%")
                else:
                    print(f"    場所: {location}, データ: {value}%")

    """ # 病害率データのヒストグラムを描画
    plot_disease_rate_histogram(desease_data2, variety_name="ターザン", use_average=True, bins=15)
    plot_disease_rate_histogram(desease_data2, variety_name="ターザン", use_average=False, bins=15)

    # 外れ値の計算
    outliers_all_years1 = calculate_outliers(desease_data2, use_average=False, method="IQR")
    
    outliers_all_years2 = calculate_outliers(desease_data2, use_average=False, method="Z-score")
    

    
    print(outliers_all_years1)
    print("--------------------------------------")
    print(outliers_all_years2) """


    # 結果を表示
    """ print("\nDesease Data:")
    for variety, years_data in desease_data1.items():
            print("--------------------------------------")
            print(f"品種名: {variety}")
            for year, locations_data in years_data.items():
                print(f"年度: {year}")
                for location, data in locations_data:
                    if location == '平均':
                        print(f"    平均値: {data:.2f}%")
                    else:
                        # print(f"    場所: {location}, 発病率: {data:.2f}%")  
                        continue


    print("\nDesease Data:")
    for variety, years_data in desease_data2.items():
            print("--------------------------------------")
            print(f"品種名: {variety}")
            for year, locations_data in years_data.items():
                print(f"年度: {year}")
                for location, data in locations_data:
                    if location == '平均':
                        print(f"    平均値: {data:.2f}%")
                    else:
                        # print(f"    場所: {location}, 発病率: {data:.2f}%")  
                        continue """