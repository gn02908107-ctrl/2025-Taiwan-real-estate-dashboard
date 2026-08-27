"""
線性迴歸 vs 隨機森林 模型比較報告

只做兩件事：
1. 用 model_comparison_v2.csv 裡「公平對照組」（同樣 v2 完整特徵，各自的正式模型設定）
   整理出比較表。
2. 針對這組公平對照，重新訓練出兩個模型的 pipeline，畫出：
   - 線性迴歸係數 vs 隨機森林特徵重要性（並排對照）
   - 實際值 vs 預測值散佈圖（並排對照）
   並輸出一份 regression_report.md 把表格、圖、結論整理成正式報告。

依賴 regression_model_v2.py 已經驗證過的資料清洗/特徵工程邏輯，這裡重新實作一份
輕量版（只練兩個模型、不做 v1 baseline、不做超參數搜尋），避免整份重跑要花好幾分鐘。
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
from sklearn.metrics import r2_score
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
# 預售屋 RF 維持原設定（未調參）
MODEL_CONFIGS = {
    "中古屋": {"use_age": True, "rf_params": {"min_samples_leaf": 5, "max_depth": None}},
    "預售屋": {"use_age": False, "rf_params": {"min_samples_leaf": 1, "max_depth": None}},
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

    fitted[house_type] = {
        "lr_pipe": lr_pipe, "lr_pred": lr_pred,
        "rf_pipe": rf_pipe, "rf_pred": rf_pred,
        "y_test": y_test, "y_train": y_train,
        "lr_r2": r2_score(y_test, lr_pred),
        "rf_r2": r2_score(y_test, rf_pred),
    }
    print(f"{house_type}：線性迴歸 R2={fitted[house_type]['lr_r2']:.3f}，"
          f"隨機森林 R2={fitted[house_type]['rf_r2']:.3f}")


# ------------------------------------------------------------
# 1. 公平對照表（從 model_comparison_v2.csv 篩選出來）
# ------------------------------------------------------------
FAIR_PAIRS = [
    ("中古屋-v2（完整特徵）", "線性迴歸"),
    ("中古屋-v2調參後（正式模型）", "隨機森林（min_samples_leaf=5）"),
    ("預售屋-v2（完整特徵）", "線性迴歸"),
    ("預售屋-v2（完整特徵）", "隨機森林"),
]

full_comparison = pd.read_csv("model_comparison_v2.csv", encoding="utf-8-sig")
mask = full_comparison.apply(lambda r: (r["版本"], r["模型"]) in FAIR_PAIRS, axis=1)
fair_table = full_comparison[mask].copy()
fair_table.insert(0, "房屋類型", fair_table["版本"].str.extract(r"^(中古屋|預售屋)"))
fair_table.insert(1, "模型類型", fair_table["模型"].apply(lambda m: "線性迴歸" if "線性" in m else "隨機森林"))
fair_table = fair_table.sort_values(["房屋類型", "模型類型"], ascending=[True, False]).reset_index(drop=True)

print("\n[公平對照表]")
print(fair_table.to_string(index=False))
fair_table.to_csv("report_fair_comparison.csv", index=False, encoding="utf-8-sig")


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

    importances = info["rf_pipe"].named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"特徵": feat_names, "重要性": importances})
    imp_df = imp_df.sort_values("重要性", ascending=False).head(12).sort_values("重要性")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].barh(coef_df["特徵"], coef_df["係數"], color=coef_colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("係數（對單價_萬元每坪的影響，萬元/坪）")
    axes[0].set_title(f"線性迴歸係數 Top 12（{house_type}）")

    axes[1].barh(imp_df["特徵"], imp_df["重要性"], color="#065A82")
    axes[1].set_xlabel("特徵重要性")
    axes[1].set_title(f"隨機森林特徵重要性 Top 12（{house_type}）")

    fig.suptitle(f"{house_type}：線性迴歸係數 vs 隨機森林特徵重要性", fontsize=13)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

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


used_lr, used_rf = fmt_row("中古屋", "線性迴歸"), fmt_row("中古屋", "隨機森林")
pre_lr, pre_rf = fmt_row("預售屋", "線性迴歸"), fmt_row("預售屋", "隨機森林")

report = f"""# 線性迴歸 vs 隨機森林：模型比較報告

資料範圍：全國 22 縣市，v2 完整特徵集（地區、車位、季度、坪數、建物型態、格局、樓層資訊；
中古屋額外加入屋齡，預售屋因屋齡結構性全缺而不採用）。以下為「公平對照」——
兩個模型使用完全相同的特徵集，各自套用其正式定案的超參數設定。

## 1. 公平對照表

| 房屋類型 | 模型 | 測試 R² | RMSE | MAE |
|---|---|---|---|---|
| 中古屋 | 線性迴歸 | {used_lr['測試R2']:.3f} | {used_lr['RMSE']:.2f} | {used_lr['MAE']:.2f} |
| 中古屋 | 隨機森林（調參後，min_samples_leaf=5） | {used_rf['測試R2']:.3f} | {used_rf['RMSE']:.2f} | {used_rf['MAE']:.2f} |
| 預售屋 | 線性迴歸 | {pre_lr['測試R2']:.3f} | {pre_lr['RMSE']:.2f} | {pre_lr['MAE']:.2f} |
| 預售屋 | 隨機森林（原設定） | {pre_rf['測試R2']:.3f} | {pre_rf['RMSE']:.2f} | {pre_rf['MAE']:.2f} |

（完整資料另存於 [report_fair_comparison.csv](report_fair_comparison.csv)）

## 2. 特徵影響力：線性迴歸係數 vs 隨機森林特徵重要性

![中古屋](report_coef_vs_importance_中古屋.png)
![預售屋](report_coef_vs_importance_預售屋.png)

線性迴歸的係數圖呈現「方向 + 大小」：藍色代表正向影響（該特徵值越大/該類別存在時，單價越高），
紅色代表負向影響。要留意的是，`行政區` 這類高基數類別經過 One-Hot 展開後彼此高度共線，
個別係數的精確數值會不穩定，解讀時應著重「方向」與「相對排序」，不宜當成嚴謹的邊際效應估計。
隨機森林的特徵重要性則不分方向，只呈現「這個特徵對降低預測誤差的貢獻有多大」。

## 3. 實際值 vs 預測值

![中古屋](report_actual_vs_pred_中古屋.png)
![預售屋](report_actual_vs_pred_預售屋.png)

紅色虛線是完美預測（預測值=實際值）。隨機森林的點明顯更貼近對角線，線性迴歸的點在單價偏高
的區間（豪宅、特殊地段）系統性地偏離對角線下方——代表線性迴歸低估了高價物件。

## 4. 線性迴歸的預測範圍問題：可能出現不合理負值

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

## 5. 結論：為什麼隨機森林表現較好

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
"""

with open("regression_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n報告已輸出：regression_report.md")
