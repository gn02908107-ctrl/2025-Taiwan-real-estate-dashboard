"""
線性迴歸 vs 隨機森林 vs XGBoost 模型比較報告

只做兩件事：
1. 用 model_comparison_v2.csv 裡「公平對照組」（同樣 v2 完整特徵，各自的正式模型設定），
   加上這裡直接訓練的 XGBoost（調參後），整理出三個模型的比較表。
2. 針對這組公平對照，重新訓練出三個模型的 pipeline，畫出：
   - 線性迴歸係數 vs 隨機森林 vs XGBoost 特徵重要性（並排對照）
   - 實際值 vs 預測值散佈圖（並排對照）
   並輸出一份 regression_report.md 把表格、圖、XGBoost 調參歷程、結論整理成正式報告。

依賴 regression_model_v2.py 已經驗證過的資料清洗/特徵工程邏輯，這裡重新實作一份
輕量版，不做 v1 baseline、不重新搜尋超參數（隨機森林、XGBoost 的最終超參數都是已經
在 regression_model_v2.py / regression_xgboost.py 搜尋定案的結果，這裡直接沿用重新訓練），
避免整份重跑要花好幾分鐘。
"""
import re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

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

# 公平對照組的正式模型設定：中古屋 RF 用調參後的 min_samples_leaf=5，
# 預售屋 RF 維持原設定（未調參）。
# xgb_params 是 regression_xgboost.py 用 RandomizedSearchCV 搜尋出來的最終參數
# （調參歷程見 regression_report.md 第 2 節：窄範圍→擴大範圍過擬合→加大 n_iter 沒解決→
# 改用「交叉驗證訓練/測試差距 ≤0.1 篩選再選最高分」規則才解決），這裡直接沿用結果重新
# 訓練最終模型，不重跑一次搜尋。
MODEL_CONFIGS = {
    "中古屋": {
        "use_age": True,
        "rf_params": {"min_samples_leaf": 5, "max_depth": None},
        "xgb_params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.2,
            "subsample": 1.0, "colsample_bytree": 0.6, "min_child_weight": 10,
        },
    },
    "預售屋": {
        "use_age": False,
        "rf_params": {"min_samples_leaf": 1, "max_depth": None},
        "xgb_params": {
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.2,
            "subsample": 0.6, "colsample_bytree": 1.0, "min_child_weight": 3,
        },
    },
}

fitted = {}

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

    lr_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)

    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, **cfg["rf_params"])),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    joblib.dump(rf_pipe, f"model_{house_type}_隨機森林.pkl", compress=3)

    xgb_pipe = Pipeline([
        ("prep", preprocess),
        ("model", XGBRegressor(**cfg["xgb_params"])),
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_pred = xgb_pipe.predict(X_test)
    xgb_pred_train = xgb_pipe.predict(X_train)

    lr_pred_train = lr_pipe.predict(X_train)
    rf_pred_train = rf_pipe.predict(X_train)

    fitted[house_type] = {
        "lr_pipe": lr_pipe, "lr_pred": lr_pred,
        "rf_pipe": rf_pipe, "rf_pred": rf_pred,
        "xgb_pipe": xgb_pipe, "xgb_pred": xgb_pred,
        "y_test": y_test, "y_train": y_train,
        "lr_r2": r2_score(y_test, lr_pred),
        "lr_r2_train": r2_score(y_train, lr_pred_train),
        "rf_r2": r2_score(y_test, rf_pred),
        "rf_r2_train": r2_score(y_train, rf_pred_train),
        "xgb_r2": r2_score(y_test, xgb_pred),
        "xgb_r2_train": r2_score(y_train, xgb_pred_train),
    }
    print(f"{house_type}：線性迴歸 R2={fitted[house_type]['lr_r2']:.3f}，"
          f"隨機森林 R2={fitted[house_type]['rf_r2']:.3f}，"
          f"XGBoost（調參後）R2={fitted[house_type]['xgb_r2']:.3f}")


# ------------------------------------------------------------
# 1. 公平對照表——三個模型都直接用這次剛訓練好的 pipeline 現算指標，
#    不依賴 model_comparison_v2.csv（那份是 regression_model_v2.py 產生的，
#    這裡若讀它，台東縣資料清乾淨後只有 XGBoost 會反映最新結果，
#    線性迴歸/隨機森林卻還是舊資料的數字，造成同一張表新舊資料混雜）。
# ------------------------------------------------------------
MODEL_ROW_SPECS = [
    ("線性迴歸", "lr", lambda ht: f"{ht}-v2（完整特徵）", lambda ht: "線性迴歸"),
    ("隨機森林", "rf",
     lambda ht: f"{ht}-v2調參後（正式模型）" if ht == "中古屋" else f"{ht}-v2（完整特徵）",
     lambda ht: "隨機森林（min_samples_leaf=5）" if ht == "中古屋" else "隨機森林"),
    ("XGBoost調參後", "xgb", lambda ht: f"{ht}-v2調參後（XGBoost）", lambda ht: "XGBoost（調參後）"),
]

fair_rows = []
for house_type in MODEL_CONFIGS:
    info = fitted[house_type]
    for model_type, key, version_fn, model_fn in MODEL_ROW_SPECS:
        r2_train = info[f"{key}_r2_train"]
        r2_test = info[f"{key}_r2"]
        pred = info[f"{key}_pred"]
        rmse = np.sqrt(mean_squared_error(info["y_test"], pred))
        mae = mean_absolute_error(info["y_test"], pred)
        fair_rows.append({
            "房屋類型": house_type, "模型類型": model_type,
            "版本": version_fn(house_type), "模型": model_fn(house_type),
            "訓練R2": round(r2_train, 3), "測試R2": round(r2_test, 3),
            "R2差距": round(r2_train - r2_test, 3), "RMSE": round(rmse, 2), "MAE": round(mae, 2),
        })

fair_table = pd.DataFrame(fair_rows)
model_order = {"線性迴歸": 0, "隨機森林": 1, "XGBoost調參後": 2}
fair_table["_順序"] = fair_table["模型類型"].map(model_order)
fair_table = fair_table.sort_values(["房屋類型", "_順序"]).drop(columns="_順序").reset_index(drop=True)

print("\n[公平對照表]")
print(fair_table.to_string(index=False))

# 寫回 report_fair_comparison.csv 時用「版本+模型」當 key 做 upsert：
# regression_xgboost.py 另外寫進去的「XGBoost（預設參數）」兩列不在這裡的計算範圍內，
# upsert 可以保留它們，只更新/新增這裡算出來的列，不會整份覆蓋掉。
key_cols = ["版本", "模型"]
try:
    existing_csv = pd.read_csv("report_fair_comparison.csv", encoding="utf-8-sig")
    new_keys = set(map(tuple, fair_table[key_cols].values))
    existing_filtered = existing_csv[~existing_csv[key_cols].apply(tuple, axis=1).isin(new_keys)]
    csv_out = pd.concat([existing_filtered, fair_table], ignore_index=True)
except FileNotFoundError:
    csv_out = fair_table

csv_out.to_csv("report_fair_comparison.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# 2. 線性迴歸係數 vs 隨機森林特徵重要性
# ------------------------------------------------------------
def get_feature_names(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    return pd.Index(names).str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)


def plot_coef_vs_importance(house_type):
    info = fitted[house_type]
    feat_names = get_feature_names(info["lr_pipe"])

    coefs = info["lr_pipe"].named_steps["model"].coef_
    coef_df = pd.DataFrame({"特徵": feat_names, "係數": coefs})
    coef_df["絕對值"] = coef_df["係數"].abs()
    coef_df = coef_df.sort_values("絕對值", ascending=False).head(12).sort_values("係數")
    coef_colors = ["#B33A3A" if v < 0 else "#065A82" for v in coef_df["係數"]]

    rf_importances = info["rf_pipe"].named_steps["model"].feature_importances_
    rf_imp_df = pd.DataFrame({"特徵": feat_names, "重要性": rf_importances})
    rf_imp_df = rf_imp_df.sort_values("重要性", ascending=False).head(12).sort_values("重要性")

    xgb_importances = info["xgb_pipe"].named_steps["model"].feature_importances_
    xgb_imp_df = pd.DataFrame({"特徵": feat_names, "重要性": xgb_importances})
    xgb_imp_df = xgb_imp_df.sort_values("重要性", ascending=False).head(12).sort_values("重要性")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    axes[0].barh(coef_df["特徵"], coef_df["係數"], color=coef_colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("係數（對單價_萬元每坪的影響，萬元/坪）")
    axes[0].set_title(f"線性迴歸係數 Top 12（{house_type}）")

    axes[1].barh(rf_imp_df["特徵"], rf_imp_df["重要性"], color="#065A82")
    axes[1].set_xlabel("特徵重要性")
    axes[1].set_title(f"隨機森林特徵重要性 Top 12（{house_type}）")

    axes[2].barh(xgb_imp_df["特徵"], xgb_imp_df["重要性"], color="#2E8B57")
    axes[2].set_xlabel("特徵重要性")
    axes[2].set_title(f"XGBoost 特徵重要性 Top 12（{house_type}）")

    fig.suptitle(f"{house_type}：線性迴歸係數 vs 隨機森林 vs XGBoost 特徵重要性", fontsize=13)
    plt.tight_layout()
    fname = f"report_coef_vs_importance_{house_type}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"圖表已輸出：{fname}")


# ------------------------------------------------------------
# 3. 實際值 vs 預測值 散佈圖
# ------------------------------------------------------------
def plot_actual_vs_pred(house_type):
    info = fitted[house_type]
    y_test = info["y_test"]
    lo, hi = y_test.min(), y_test.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].scatter(y_test, info["lr_pred"], alpha=0.08, s=8, color="#065A82", edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], color="#B33A3A", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("實際單價（萬元/坪）")
    axes[0].set_ylabel("預測單價（萬元/坪）")
    axes[0].set_title(f"線性迴歸（R²={info['lr_r2']:.3f}）")

    axes[1].scatter(y_test, info["rf_pred"], alpha=0.08, s=8, color="#065A82", edgecolors="none")
    axes[1].plot([lo, hi], [lo, hi], color="#B33A3A", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("實際單價（萬元/坪）")
    axes[1].set_ylabel("預測單價（萬元/坪）")
    axes[1].set_title(f"隨機森林（R²={info['rf_r2']:.3f}）")

    axes[2].scatter(y_test, info["xgb_pred"], alpha=0.08, s=8, color="#065A82", edgecolors="none")
    axes[2].plot([lo, hi], [lo, hi], color="#B33A3A", linewidth=1.2, linestyle="--")
    axes[2].set_xlabel("實際單價（萬元/坪）")
    axes[2].set_ylabel("預測單價（萬元/坪）")
    axes[2].set_title(f"XGBoost（R²={info['xgb_r2']:.3f}）")

    fig.suptitle(f"{house_type}：實際值 vs 預測值（紅色虛線＝完美預測）", fontsize=13)
    plt.tight_layout()
    fname = f"report_actual_vs_pred_{house_type}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"圖表已輸出：{fname}")


for house_type in MODEL_CONFIGS:
    plot_coef_vs_importance(house_type)
    plot_actual_vs_pred(house_type)


# ------------------------------------------------------------
# 4. 中古屋線性迴歸負值預測檢查
#    線性迴歸擬合的是一個無邊界的超平面，沒有任何機制保證輸出 >= 0；
#    隨機森林的預測值是訓練資料裡「葉節點內實際觀察到的目標值」取平均，
#    平均值的範圍必然落在訓練集觀察到的最小值與最大值之間，不可能是負的。
# ------------------------------------------------------------
used_info = fitted["中古屋"]
lr_pred_used = used_info["lr_pred"]
rf_pred_used = used_info["rf_pred"]
y_train_used = used_info["y_train"]

neg_mask = lr_pred_used < 0
neg_count = int(neg_mask.sum())
neg_total = len(lr_pred_used)
neg_ratio = neg_count / neg_total

rf_pred_min, rf_pred_max = rf_pred_used.min(), rf_pred_used.max()
train_min, train_max = y_train_used.min(), y_train_used.max()

print(f"\n中古屋線性迴歸負值預測：{neg_count} / {neg_total} 筆（{neg_ratio:.4%}）")
print(f"隨機森林預測值範圍：[{rf_pred_min:.2f}, {rf_pred_max:.2f}]，"
      f"訓練集實際單價範圍：[{train_min:.2f}, {train_max:.2f}]")


# ------------------------------------------------------------
# 5. 寫成正式報告 regression_report.md
# ------------------------------------------------------------
def fmt_row(house_type, model_label):
    r = fair_table[(fair_table["房屋類型"] == house_type) & (fair_table["模型類型"] == model_label)].iloc[0]
    return r


used_lr, used_rf, used_xgb = fmt_row("中古屋", "線性迴歸"), fmt_row("中古屋", "隨機森林"), fmt_row("中古屋", "XGBoost調參後")
pre_lr, pre_rf, pre_xgb = fmt_row("預售屋", "線性迴歸"), fmt_row("預售屋", "隨機森林"), fmt_row("預售屋", "XGBoost調參後")

report = f"""# 線性迴歸 vs 隨機森林 vs XGBoost：模型比較報告

資料範圍：全國 21 縣市，v2 完整特徵集（地區、車位、季度、坪數、建物型態、格局、樓層資訊；
中古屋額外加入屋齡，預售屋因屋齡結構性全缺而不採用）。以下為「公平對照」——
三個模型使用完全相同的特徵集，各自套用其正式定案的超參數設定。

## 資料附註：台東縣命名重複問題（已於 2026-08-30 修正）

**發現過程**：本報告初版開頭寫「全國 22 縣市」，是直接採用 `data['縣市'].nunique()` 的結果。
但台灣行政區劃裡這份資料實際涵蓋的縣市只有 21 個（未涵蓋連江縣），22 這個數字啟人疑竇，
進一步比對後發現資料庫縣市欄位裡「台東縣」與「臺東縣」被當成兩個不同字串。

**驗證過程**：查詢資料表名稱後發現，S1~S3 的台東縣資料表一律用「臺東縣」，只有 S4
同時存在「台東縣」與「臺東縣」兩組、共 8 張表（不含車位／含車位 × 中古屋／預售屋）。
逐一比對這 4 對表格的筆數與內容（`DataFrame.equals()`），確認完全 byte-for-byte 相同——
不是兩批不同的交易紀錄，是同一批資料被存了兩次，只是縣市欄位用字不同。以此為契機，
進一步把「台→臺」正規化後比對全部 332 張表的縣市字串，確認台北市、台中市、台南市等
其他縣市都沒有類似的寫法不一致問題，台東縣是單一事件。

**修正方式**：備份原始資料庫後，刪除 S4 底下 4 張用「台東縣」命名的重複表格，只保留
跟 S1~S3 一致的「臺東縣」版本，並執行 `VACUUM` 回收空間。修正後
`data['縣市'].nunique()` 正確等於 21，資料表總數從 336 降為 332，S4 台東縣的交易筆數
沒有任何流失。本報告與其餘產出物（`report_fair_comparison.csv`、兩個 `.pkl` 模型檔、
下方所有圖表）皆已用修正後的資料重新訓練——train_test_split 的 `random_state=42`
維持不變，唯一改變的變數是資料本身變乾淨了，不會混入其他變因。

## 1. 公平對照表

| 房屋類型 | 模型 | 測試 R² | RMSE | MAE |
|---|---|---|---|---|
| 中古屋 | 線性迴歸 | {used_lr['測試R2']:.3f} | {used_lr['RMSE']:.2f} | {used_lr['MAE']:.2f} |
| 中古屋 | 隨機森林（調參後，min_samples_leaf=5） | {used_rf['測試R2']:.3f} | {used_rf['RMSE']:.2f} | {used_rf['MAE']:.2f} |
| 中古屋 | XGBoost（調參後） | {used_xgb['測試R2']:.3f} | {used_xgb['RMSE']:.2f} | {used_xgb['MAE']:.2f} |
| 預售屋 | 線性迴歸 | {pre_lr['測試R2']:.3f} | {pre_lr['RMSE']:.2f} | {pre_lr['MAE']:.2f} |
| 預售屋 | 隨機森林（原設定） | {pre_rf['測試R2']:.3f} | {pre_rf['RMSE']:.2f} | {pre_rf['MAE']:.2f} |
| 預售屋 | XGBoost（調參後） | {pre_xgb['測試R2']:.3f} | {pre_xgb['RMSE']:.2f} | {pre_xgb['MAE']:.2f} |

（完整資料另存於 [report_fair_comparison.csv](report_fair_comparison.csv)）

## 2. XGBoost 超參數調整歷程

XGBoost 的調參不是一次到位，過程本身比最終數字更值得記錄（完整程式碼見
[regression_xgboost.py](regression_xgboost.py)）：

> **時間點提醒**：第 1~3 步的所有數字，都是在台東縣命名重複問題（見上方「資料附註」）
> 修正**之前**、用當時還沒清乾淨的資料調參時測得的紀錄，目的是記錄「怎麼發現交叉驗證選贏家
> 會選到過擬合解」這個方法論探索過程，**不代表目前正式模型的表現**。這幾步的數字彼此之間
> （用的都是同一份舊資料）互相比較是公平的，但不能拿來跟第 1 節公平對照表的數字比較——
> 那是修正後的資料重新訓練出來的，比較基準不同。

1. **窄範圍搜尋**（`n_estimators`∈{{100,200,300}}、`max_depth`∈{{3,4,5,6}}、`learning_rate`∈{{0.05,0.1,0.2}}、
   `subsample`/`colsample_bytree`∈{{0.6,0.8,1.0}}、`min_child_weight`∈{{1,3,5,10}}，
   `RandomizedSearchCV`、`n_iter=10`、`cv=3`）：中古屋測試 R²=0.740、差距 0.044；
   預售屋測試 R²=0.968、差距 0.008。兩者都通過 ≤0.1 的過擬合門檻，看起來很順利。

2. **擴大搜尋範圍**（`n_estimators`加入 500、`max_depth`加入 8、`learning_rate`加入 0.01，其餘不變）之後，
   中古屋測試 R² 反而掉到 0.705，訓練/測試差距暴增到 0.167，觸發過擬合警訊；預售屋則沒事
   （0.967、差距 0.010）。搜出來的中古屋贏家參數是 `max_depth=8`、`min_child_weight=1`、
   `subsample=0.6`——樹更深、正則化更弱、抽樣隨機性更大，剛好是最容易過擬合的方向。

3. **把 `n_iter` 從 10 加到 20**，範圍不變，中古屋的結果完全沒變——因為 `random_state=42` 固定，
   前 10 組抽樣序列跟上次相同，多抽的 10 組沒有一組贏過原本那組候選的交叉驗證分數，證明問題不在
   「抽太少」，是選贏家的規則本身有問題。預售屋這邊倒是找到更好的組合（測試 R²=0.978、差距 0.015）。

4. **改變選贏家的規則**：不再直接用 `RandomizedSearchCV.best_params_`（單純看交叉驗證平均測試分數），
   改成加上 `return_train_score=True`，從 `cv_results_` 算出每組候選「交叉驗證訓練分數－測試分數」，
   篩選出差距 ≤0.1 的候選，再從中挑測試分數最高的一組。套用這個規則後，中古屋選出
   `max_depth=5`、`min_child_weight=10`、`subsample=1.0` 這組——在當時同一份舊資料上，
   過擬合差距從第 2 步的 0.167 降到 0.045，問題解決；預售屋維持第 3 步找到的組合不變
   （它本來就滿足新規則）。這兩組參數就是目前正式採用的 XGBoost 設定，後續已經用修正後的
   21 縣市資料重新訓練，實際測試 R² 與差距請見第 1 節公平對照表（中古屋
   {used_xgb['測試R2']:.3f}／預售屋 {pre_xgb['測試R2']:.3f}）。

**這個過程暴露的重點發現**：`RandomizedSearchCV` 預設用交叉驗證平均測試分數選贏家，這個分數只看
「準不準」，不看「訓練分數跟測試分數差多少」。一組參數可以在交叉驗證的平均測試分數上表現不錯，
同時訓練分數又特別高（代表它在記憶訓練資料），這種組合的平均分數可能剛好贏過其他組，卻是用過擬合
換來的分數，實際泛化能力並不可靠。要避免選到這種解，篩選規則必須把「差距」也當成篩選條件，
不能只看單一分數排名。

## 3. 特徵影響力：線性迴歸係數 vs 隨機森林 vs XGBoost 特徵重要性

![中古屋](report_coef_vs_importance_中古屋.png)
![預售屋](report_coef_vs_importance_預售屋.png)

線性迴歸的係數圖呈現「方向 + 大小」：藍色代表正向影響（該特徵值越大/該類別存在時，單價越高），
紅色代表負向影響。要留意的是，`行政區` 這類高基數類別經過 One-Hot 展開後彼此高度共線，
個別係數的精確數值會不穩定，解讀時應著重「方向」與「相對排序」，不宜當成嚴謹的邊際效應估計。
隨機森林與 XGBoost 的特徵重要性則不分方向，只呈現「這個特徵對降低預測誤差的貢獻有多大」——
但兩者算重要性的方式不同（隨機森林用不純度下降量，XGBoost 預設用分裂增益），排序不必然一致：
以中古屋為例，隨機森林的 Top 12 混了屋齡、總樓層數、總坪數三個數值特徵，XGBoost 的 Top 12
卻清一色是行政區類別，數值特徵完全沒擠進榜單。這不代表 XGBoost 忽略了屋齡或坪數的訊號
（兩者測試 R² 很接近就是反證），只是這兩種重要性指標本來就不是同一把尺，不宜直接拿排序做比較。

## 4. 實際值 vs 預測值

![中古屋](report_actual_vs_pred_中古屋.png)
![預售屋](report_actual_vs_pred_預售屋.png)

紅色虛線是完美預測（預測值=實際值）。隨機森林、XGBoost 的點都明顯比線性迴歸更貼近對角線，
線性迴歸的點在單價偏高的區間（豪宅、特殊地段）系統性地偏離對角線下方——代表線性迴歸低估了
高價物件。隨機森林與 XGBoost 兩者的散佈圖非常接近，肉眼較難分辨優劣，數字上的差異要看表格。

## 5. 線性迴歸的預測範圍問題：可能出現不合理負值

實際檢查中古屋線性迴歸模型在測試集上的預測值（`lr_pred`），發現有 **{neg_count} 筆／共 {neg_total} 筆
（{neg_ratio:.4%}）預測出負的單價**——但「單價_萬元每坪」在真實世界裡不可能是負數，這是模型的明顯瑕疵。

同一份測試集上，隨機森林的預測值範圍是 **[{rf_pred_min:.2f}, {rf_pred_max:.2f}]**，
對照訓練集實際觀察到的單價範圍 **[{train_min:.2f}, {train_max:.2f}]**，隨機森林沒有任何一筆負值。

這不是巧合，而是兩種模型的機制差異所致：

- **線性迴歸擬合的是一個沒有邊界的超平面**：`預測值 = 截距 + Σ(係數 × 特徵值)`。當某筆資料的特徵組合
  落在訓練資料較稀疏、或多個負向係數的類別疊加在一起時（例如低價地區 × 高屋齡 × 小坪數的組合），
  加總結果可以跌到 0 以下。線性迴歸完全不知道「單價不能是負的」這個常識，因為這個限制從未寫進模型裡。

- **隨機森林的預測值必然落在訓練資料的觀察範圍內**：每一棵樹的預測是某個葉節點裡「實際訓練樣本的
  平均值」，森林的最終預測再對這些樹取平均。既然平均值不可能超出被平均的那些數字的範圍，
  隨機森林的預測天生就被訓練資料夾在 [{train_min:.2f}, {train_max:.2f}] 之間，不會外插到不合理的負值，
  但代價是它也無法預測出訓練資料範圍以外的極端值（例如史上最高單價的物件）。

## 6. 結論：為什麼隨機森林表現較好

中古屋測試 R² 從線性迴歸的 {used_lr['測試R2']:.3f} 提升到隨機森林的 {used_rf['測試R2']:.3f}，
預售屋從 {pre_lr['測試R2']:.3f} 提升到 {pre_rf['測試R2']:.3f}。差異的根源在於兩個模型對「特徵如何影響房價」
的假設不同：

- **線性迴歸假設每個特徵對房價的影響是一條直線、而且彼此獨立**。例如它認為「每增加一坪」對單價的
  加成是固定的，不管房子在哪個行政區、屋齡多大；也假設地區的影響和坪數的影響可以直接相加，
  不會互相干擾。但真實房價明顯不是這樣：市中心蛋黃區的坪數溢價，跟郊區的坪數溢價幅度完全不同
  （這是「特徵交互作用」）；屋齡對房價的影響也不是等速遞減，屋齡 0~10 年的跌價速度，
  跟屋齡 30~40 年的老屋通常不一樣（這是「非線性關係」）。這些線性迴歸都學不到。

- **隨機森林用一堆決策樹分裂資料**，每一次分裂都可以先看地區、再看坪數、再看屋齡，天生就能表達
  「在信義區，坪數的影響比較大」這種交互作用，也能表達「屋齡在某個轉折點後跌價速度變快」這種
  非線性曲線，而不需要事先假設任何函數形式。

也因為如此，這個比較不只是「哪個模型數字比較漂亮」，而是反映了房價本身的資料生成機制：
地區、屋況、格局之間彼此牽動、影響幅度隨情境而變——這正是樹模型的強項，也是線性迴歸的天生限制。

那 XGBoost 呢？同樣是樹模型，理論上該有跟隨機森林差不多的能力抓非線性關係與特徵交互作用，
實際結果也確實如此：兩個房屋類型，調參後的 XGBoost（中古屋 {used_xgb['測試R2']:.3f}、
預售屋 {pre_xgb['測試R2']:.3f}）都只些微落後隨機森林（中古屋 {used_rf['測試R2']:.3f}、
預售屋 {pre_rf['測試R2']:.3f}），差距都在 0.01 左右，不是量級上的差距。這說明當初把隨機森林
定為正式模型是合理的判斷——不是因為 XGBoost 不好，而是在這份資料、這組特徵下，兩種樹模型的
表現本來就非常接近，隨機森林剛好略勝一籌，且已經完成過擬合檢查與調參，沒有必要為了微小差距
再引入一個新的模型依賴。
"""

with open("regression_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n報告已輸出：regression_report.md")
