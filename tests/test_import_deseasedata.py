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

            organized_data[variety][year].append(("avg", value))
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


if __name__ == '__main__':

    # 使用例
    syukaku_data_path = 'resources/desease_data/disease_data_syukaku.xlsx'
    tyozou_data_path = "resources/desease_data/disease_data_tyozou.xlsx"
    start_year = 1990
    end_year = 2023
    target_varieties = ['ターザン']  # 取得したい品種を指定

    # Example usage
    syukaku_desease_data = import_desease_data(syukaku_data_path, start_year=start_year, end_year=end_year, verbose=True)
    tyozou_desease_data = import_desease_data(tyozou_data_path, start_year=start_year, end_year=end_year, verbose=True)
    
    # Print the output
    for variety, years_data in syukaku_desease_data.items():
        print(f"\n品種名: {variety}")
        for year, locations_data in years_data.items():
            print(f"  年度: {year}")
            for location, value in locations_data:
                if location == "avg":
                    print(f"    平均値: {value}%")
                else:
                    print(f"    場所: {location}, データ: {value}%")

    
    for variety, years_data in tyozou_desease_data.items():
        print(f"\n品種名: {variety}")
        for year, locations_data in years_data.items():
            print(f"  年度: {year}")
            for location, value in locations_data:
                if location == "avg":
                    print(f"    平均値: {value}%")
                else:
                    print(f"    場所: {location}, データ: {value}%")