"""
XGBoost（預設參數）跟既有線性迴歸／隨機森林公平對照

只做四件事：
1. 資料讀取、特徵工程完全比照 regression_report.py（行政區、含車位、季度、總坪數、
   建物型態、格局、樓層、樓層比例；中古屋另外加屋齡），train_test_split 也用同樣的
   random_state=42，確保切出來的訓練/測試集跟既有三個模型一致，比較才公平。
2. 用 XGBoost 預設參數，分別對中古屋、預售屋訓練模型。
3. 算測試 R²、RMSE、MAE（順便也算訓練 R²，跟 report_fair_comparison.csv 既有欄位對齊）。
4. 把這兩列加進既有的 report_fair_comparison.csv，原本四列（線性迴歸／隨機森林）不動。

這次刻意不存模型檔（.pkl）、不動 regression_report.md、不碰 Dashboard——
只是先看 XGBoost 的表現，還沒決定要不要正式採用。
"""
import re
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

DB_PATH = "Database/全國房屋實價登錄資料.db"
TABLE_PATTERN = re.compile(r"^(S\d)_(.+?)房地產交易資料_(不含車位|含車位)\((中古屋|預售屋)\)$")
TARGET = "單價_萬元每坪"


def cn2int(cn):
    """樓層數字轉阿拉伯數字（同時處理中文數字文字與純數字字串，詳見 regression_model_v2.py）"""
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


data = load_all_data()
data["行政區"] = data["縣市"] + data["鄉鎮市區"]
data["總坪數"] = pd.to_numeric(data["總坪數"], errors="coerce")
data["移轉樓層"] = data["移轉層次"].apply(cn2int)
data["總樓層數_num"] = data["總樓層數"].apply(cn2int)
data["樓層比例"] = data["移轉樓層"] / data["總樓層數_num"]
data.loc[data["樓層比例"] > 1, "樓層比例"] = np.nan
data = data.dropna(subset=["單價_萬元每坪", "總坪數"])

FEATURES_V1 = ["行政區", "含車位", "季度", "總坪數"]
CATEGORICAL_V1 = ["行政區", "含車位", "季度"]
NUMERIC_V1 = ["總坪數"]
EXTRA_CATEGORICAL_V2 = ["建物型態"]
EXTRA_NUMERIC_V2 = [
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛",
    "移轉樓層", "總樓層數_num", "樓層比例",
]

MODEL_CONFIGS = {
    "中古屋": {"use_age": True},
    "預售屋": {"use_age": False},
}

xgb_rows = []

for house_type, cfg in MODEL_CONFIGS.items():
    df_type = data[data["房屋類型"] == house_type].copy()

    features = FEATURES_V1 + EXTRA_CATEGORICAL_V2 + EXTRA_NUMERIC_V2
    categorical = CATEGORICAL_V1 + EXTRA_CATEGORICAL_V2
    numeric = NUMERIC_V1 + EXTRA_NUMERIC_V2
    if cfg["use_age"]:
        features = features + ["屋齡"]
        numeric = numeric + ["屋齡"]

    X = df_type[features]
    y = df_type[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])

    xgb_pipe = Pipeline([("prep", preprocess), ("model", XGBRegressor())])
    xgb_pipe.fit(X_train, y_train)

    pred_train = xgb_pipe.predict(X_train)
    pred_test = xgb_pipe.predict(X_test)

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    gap = r2_train - r2_test

    print(f"{house_type}（XGBoost 預設參數）：訓練R2={r2_train:.3f} 測試R2={r2_test:.3f} "
          f"差距={gap:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")

    xgb_rows.append({
        "房屋類型": house_type,
        "模型類型": "XGBoost",
        "版本": f"{house_type}-v2（完整特徵）",
        "模型": "XGBoost（預設參數）",
        "訓練R2": round(r2_train, 3),
        "測試R2": round(r2_test, 3),
        "R2差距": round(gap, 3),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
    })


# ------------------------------------------------------------
# 5. XGBoost 超參數調參（比照 regression_model_v2.py 調隨機森林的做法：
#    RandomizedSearchCV、cv=3、n_iter=10），中古屋、預售屋分開搜尋。
#    調完一樣檢查訓練/測試 R² 差距，>0.1 視為過擬合警訊。
# ------------------------------------------------------------
XGB_PARAM_DIST = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5, 10],
}

tuned_rows = []

for house_type, cfg in MODEL_CONFIGS.items():
    df_type = data[data["房屋類型"] == house_type].copy()

    features = FEATURES_V1 + EXTRA_CATEGORICAL_V2 + EXTRA_NUMERIC_V2
    categorical = CATEGORICAL_V1 + EXTRA_CATEGORICAL_V2
    numeric = NUMERIC_V1 + EXTRA_NUMERIC_V2
    if cfg["use_age"]:
        features = features + ["屋齡"]
        numeric = numeric + ["屋齡"]

    X = df_type[features]
    y = df_type[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])

    search_pipe = Pipeline([("prep", preprocess), ("model", XGBRegressor())])
    search = RandomizedSearchCV(
        search_pipe, XGB_PARAM_DIST, n_iter=20, cv=3, scoring="r2",
        random_state=42, n_jobs=-1, verbose=1, return_train_score=True,
    )
    search.fit(X_train, y_train)

    # 不直接用 search.best_params_（那只看交叉驗證測試分數，不管過不過擬合）。
    # 改成：先篩出交叉驗證「訓練分數-測試分數」差距 ≤ 0.1 的候選，再從中選測試分數最高的一組。
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["cv_gap"] = cv_results["mean_train_score"] - cv_results["mean_test_score"]
    candidates = cv_results[cv_results["cv_gap"] <= 0.1]
    if candidates.empty:
        print(f"⚠ {house_type}：沒有任何候選組合的交叉驗證差距 ≤ 0.1，"
              f"退而求其次，改選全部候選裡差距最小的一組。")
        winner = cv_results.sort_values("cv_gap").iloc[0]
    else:
        winner = candidates.sort_values("mean_test_score", ascending=False).iloc[0]

    best_params = {k.replace("model__", ""): v for k, v in winner["params"].items()}
    print(f"\n[{house_type} XGBoost 調參・低過擬合優先] 最佳參數: {best_params}")
    print(f"[{house_type} XGBoost 調參・低過擬合優先] 交叉驗證：訓練R2={winner['mean_train_score']:.3f} "
          f"測試R2={winner['mean_test_score']:.3f} 差距={winner['cv_gap']:.3f}")

    best_model = XGBRegressor(**best_params)
    best_pipe = Pipeline([("prep", preprocess), ("model", best_model)])
    best_pipe.fit(X_train, y_train)
    pred_train = best_pipe.predict(X_train)
    pred_test = best_pipe.predict(X_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    gap = r2_train - r2_test

    print(f"[{house_type} XGBoost 調參後] 訓練R2={r2_train:.3f} 測試R2={r2_test:.3f} "
          f"差距={gap:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")
    if gap > 0.1:
        print(f"⚠ 過擬合警訊：{house_type} XGBoost調參後 差距達 {gap:.3f}，模型可能過擬合。")
    else:
        print(f"✓ {house_type} XGBoost調參後 過擬合檢查通過（差距 ≤ 0.1）。")

    tuned_rows.append({
        "房屋類型": house_type,
        "模型類型": "XGBoost調參後",
        "版本": f"{house_type}-v2調參後（XGBoost）",
        "模型": "XGBoost（調參後）",
        "訓練R2": round(r2_train, 3),
        "測試R2": round(r2_test, 3),
        "R2差距": round(gap, 3),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
    })


# ------------------------------------------------------------
# 把 XGBoost（預設 + 調參後）結果更新進 report_fair_comparison.csv。
# 用「版本+模型」當 key 做 upsert：跟這次結果同名的舊列會被取代（避免重跑腳本時
# 把預設 XGBoost 那兩列越疊越多次），其餘既有列（線性迴歸／隨機森林）完全不動。
# ------------------------------------------------------------
existing = pd.read_csv("report_fair_comparison.csv", encoding="utf-8-sig")
new_rows_df = pd.DataFrame(xgb_rows + tuned_rows)

key_cols = ["版本", "模型"]
new_keys = set(map(tuple, new_rows_df[key_cols].values))
existing_filtered = existing[~existing[key_cols].apply(tuple, axis=1).isin(new_keys)]

updated = pd.concat([existing_filtered, new_rows_df], ignore_index=True)

print("\n[更新後的公平對照表]")
print(updated.to_string(index=False))
updated.to_csv("report_fair_comparison.csv", index=False, encoding="utf-8-sig")
print("\nreport_fair_comparison.csv 已更新")
