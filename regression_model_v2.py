import re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體，Windows 內建
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DB_PATH = "Database/全國房屋實價登錄資料.db"
TABLE_PATTERN = re.compile(r"^(S\d)_(.+?)房地產交易資料_(不含車位|含車位)\((中古屋|預售屋)\)$")


def cn2int(cn):
    """樓層數字轉阿拉伯數字，只取逗號前第一段，無法解析回傳 None。

    中古屋（_a.csv 來源）的樓層欄位是中文數字文字（如「三層」），
    預售屋（_b.csv 來源）的「總樓層數」欄位卻是原始的阿拉伯數字，
    兩種來源格式不一致，所以兩種都要能解析。
    """
    if cn is None:
        return None
    cn = str(cn).split("，")[0].split(",")[0].replace("層", "").strip()
    if cn.isdigit():
        return int(cn)
    digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if not cn or not all(c in set("零一二三四五六七八九十") for c in cn):
        return None
    if cn == "十":
        return 10
    if "十" in cn:
        left, _, right = cn.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(cn)


def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    df_list = []
    for t in tables:
        m = TABLE_PATTERN.match(t)
        if not m:
            continue
        season, county, car_label, house_type = m.groups()
        df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
        df["季度"] = season
        df["縣市"] = county
        df["含車位"] = (car_label == "含車位")
        df["房屋類型"] = house_type
        df_list.append(df)
    conn.close()
    return pd.concat(df_list, ignore_index=True)


# ------------------------------------------------------------
# 1. 資料準備（範圍：全國）
# ------------------------------------------------------------
data = load_all_data()
data["行政區"] = data["縣市"] + data["鄉鎮市區"]  # 避免不同縣市同名行政區混淆
data["總坪數"] = pd.to_numeric(data["總坪數"], errors="coerce")
data["移轉樓層"] = data["移轉層次"].apply(cn2int)
data["總樓層數_num"] = data["總樓層數"].apply(cn2int)
data["樓層比例"] = data["移轉樓層"] / data["總樓層數_num"]
data.loc[data["樓層比例"] > 1, "樓層比例"] = np.nan  # 樓層比例不該超過1，超過視為解析異常

data = data.dropna(subset=["單價_萬元每坪", "總坪數"])

TARGET = "單價_萬元每坪"

# v1：原始 baseline 特徵
FEATURES_V1 = ["行政區", "含車位", "季度", "總坪數"]
CATEGORICAL_V1 = ["行政區", "含車位", "季度"]
NUMERIC_V1 = ["總坪數"]

# v2 額外加入的特徵（格局、樓層、建物型態；屋齡依房屋類型決定要不要加）
EXTRA_CATEGORICAL_V2 = ["建物型態"]
EXTRA_NUMERIC_V2 = [
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛",
    "移轉樓層", "總樓層數_num", "樓層比例",
]


def run_experiment(df, features, categorical, numeric, label):
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])

    results = []
    fitted = {}
    for name, model in [
        ("線性迴歸", LinearRegression()),
        ("隨機森林", RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
    ]:
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        pred_train = pipe.predict(X_train)
        r2 = r2_score(y_test, pred)
        r2_train = r2_score(y_train, pred_train)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        gap = r2_train - r2
        results.append({"版本": label, "模型": name, "訓練R2": round(r2_train, 3),
                         "測試R2": round(r2, 3), "R2差距": round(gap, 3),
                         "RMSE": round(rmse, 2), "MAE": round(mae, 2)})
        if gap > 0.1:
            print(f"⚠ 過擬合警訊：{label}／{name} 訓練R2={r2_train:.3f} 遠高於測試R2={r2:.3f}（差距{gap:.3f}），"
                  f"模型記住了訓練資料的雜訊，泛化能力可能不足。")
        fitted[name] = (pipe, pred, y_test)
    return results, fitted


def plot_importance(fitted, house_type, subtitle="v2"):
    rf_pipe, _, _ = fitted["隨機森林"]
    feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"特徵": feature_names, "重要性": importances})
    imp_df["特徵"] = imp_df["特徵"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    imp_df = imp_df.sort_values("重要性", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp_df["特徵"][::-1], imp_df["重要性"][::-1], color="#065A82")
    ax.set_xlabel("特徵重要性")
    ax.set_title(f"隨機森林特徵重要性 Top 12（{house_type}・{subtitle}）")
    plt.tight_layout()
    fname = f"feature_importance_{house_type}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"圖表已輸出：{fname}")


# ------------------------------------------------------------
# 2. 分房屋類型建模：中古屋（含屋齡） / 預售屋（不含屋齡，因為結構性全缺）
# ------------------------------------------------------------
all_results = []

for house_type, use_age in [("中古屋", True), ("預售屋", False)]:
    df_type = data[data["房屋類型"] == house_type].copy()

    features_v2 = FEATURES_V1 + EXTRA_CATEGORICAL_V2 + EXTRA_NUMERIC_V2
    categorical_v2 = CATEGORICAL_V1 + EXTRA_CATEGORICAL_V2
    numeric_v2 = NUMERIC_V1 + EXTRA_NUMERIC_V2
    if use_age:
        features_v2 = features_v2 + ["屋齡"]
        numeric_v2 = numeric_v2 + ["屋齡"]

    results_v1, _ = run_experiment(
        df_type, FEATURES_V1, CATEGORICAL_V1, NUMERIC_V1, f"{house_type}-v1（baseline）"
    )
    results_v2, fitted_v2 = run_experiment(
        df_type, features_v2, categorical_v2, numeric_v2, f"{house_type}-v2（完整特徵）"
    )
    all_results.extend(results_v1)
    all_results.extend(results_v2)

    plot_importance(fitted_v2, house_type)

comparison = pd.DataFrame(all_results)
print(comparison.to_string(index=False))
comparison.to_csv("model_comparison_v2.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# 3. 針對中古屋隨機森林的過擬合問題調參
#    用 RandomizedSearchCV 在 max_depth / min_samples_leaf 上搜尋，
#    找出「準確度跟 v2(max_depth=None) 差不多、但訓練/測試差距更小」的設定。
# ------------------------------------------------------------
def tune_random_forest_used():
    house_type = "中古屋"
    df_type = data[data["房屋類型"] == house_type].copy()
    features_v2 = FEATURES_V1 + EXTRA_CATEGORICAL_V2 + EXTRA_NUMERIC_V2 + ["屋齡"]
    categorical_v2 = CATEGORICAL_V1 + EXTRA_CATEGORICAL_V2
    numeric_v2 = NUMERIC_V1 + EXTRA_NUMERIC_V2 + ["屋齡"]

    X = df_type[features_v2]
    y = df_type[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_v2),
        ("num", SimpleImputer(strategy="median"), numeric_v2),
    ])

    # 搜尋階段先用較少的樹（150）加速，只是為了比較不同 max_depth/min_samples_leaf 組合
    search_pipe = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ])
    param_dist = {
        "model__max_depth": [10, 15, 20, 25, 30, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
    }
    search = RandomizedSearchCV(
        search_pipe, param_dist, n_iter=10, cv=3, scoring="r2",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    print("\n[調參] 最佳參數:", search.best_params_)
    print("[調參] 交叉驗證平均測試R2:", round(search.best_score_, 3))

    # 找到最佳深度/葉節點設定後，換回 200 棵樹重新訓練，做最終評估
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    final_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, **best_params)
    final_pipe = Pipeline([("prep", preprocess), ("model", final_model)])
    final_pipe.fit(X_train, y_train)

    pred_train = final_pipe.predict(X_train)
    pred_test = final_pipe.predict(X_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    gap = r2_train - r2_test

    print(f"[調參後・200棵樹] 參數={best_params} 訓練R2={r2_train:.3f} 測試R2={r2_test:.3f} "
          f"差距={gap:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")
    if gap > 0.1:
        print(f"⚠ 過擬合警訊：調參後差距仍達 {gap:.3f}，尚未改善到可接受範圍。")
    else:
        print("✓ 過擬合已改善到可接受範圍（差距 ≤ 0.1）。")

    # 這是正式採用的中古屋模型，重新輸出特徵重要性圖（取代原本 v2 過擬合版本的圖）
    fitted_final = {"隨機森林": (final_pipe, pred_test, y_test)}
    plot_importance(fitted_final, house_type, subtitle=f"調參後正式模型・min_samples_leaf={best_params['min_samples_leaf']}")

    return {
        "版本": "中古屋-v2調參後（正式模型）",
        "模型": f"隨機森林（min_samples_leaf={best_params['min_samples_leaf']}）",
        "訓練R2": round(r2_train, 3), "測試R2": round(r2_test, 3), "R2差距": round(gap, 3),
        "RMSE": round(rmse, 2), "MAE": round(mae, 2),
        "max_depth": best_params.get("max_depth"), "min_samples_leaf": best_params.get("min_samples_leaf"),
    }


tuning_result = tune_random_forest_used()

# 把調參後的結果「加一列」進實驗紀錄，保留調參前的紀錄不覆蓋——
# 調參前後的對比本身就是素材，過程不該因為選定正式版本就被刪掉。
comparison_row = {k: tuning_result[k] for k in ["版本", "模型", "訓練R2", "測試R2", "R2差距", "RMSE", "MAE"]}
comparison = pd.concat([comparison, pd.DataFrame([comparison_row])], ignore_index=True)
print("\n[更新後的完整實驗紀錄]")
print(comparison.to_string(index=False))
comparison.to_csv("model_comparison_v2.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# 4. 正式模型參數總結（履歷/面試用，跟原始實驗紀錄分開存，只放最終定案的設定）
# ------------------------------------------------------------
presale_row = next(
    r for r in all_results if r["版本"] == "預售屋-v2（完整特徵）" and r["模型"] == "隨機森林"
)

final_summary = pd.DataFrame([
    {
        "房屋類型": "中古屋", "模型": "隨機森林",
        "n_estimators": 200, "max_depth": "None",
        "min_samples_leaf": tuning_result["min_samples_leaf"],
        "特徵集": "v2（含屋齡）",
        "訓練R2": tuning_result["訓練R2"], "測試R2": tuning_result["測試R2"],
        "R2差距": tuning_result["R2差距"], "RMSE": tuning_result["RMSE"], "MAE": tuning_result["MAE"],
    },
    {
        "房屋類型": "預售屋", "模型": "隨機森林",
        "n_estimators": 200, "max_depth": "None",
        "min_samples_leaf": 1,
        "特徵集": "v2（不含屋齡，屋齡結構性全缺）",
        "訓練R2": presale_row["訓練R2"], "測試R2": presale_row["測試R2"],
        "R2差距": presale_row["R2差距"], "RMSE": presale_row["RMSE"], "MAE": presale_row["MAE"],
    },
])
print("\n[正式模型參數總結]")
print(final_summary.to_string(index=False))
final_summary.to_csv("final_model_summary.csv", index=False, encoding="utf-8-sig")
print("\n總結表已輸出：final_model_summary.csv")
