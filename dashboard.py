import re
import sqlite3
import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np

# ------------------------------------------------------------
# 頁面基本設定
# ------------------------------------------------------------
st.set_page_config(page_title="全國房地產交易分析", layout="wide")
st.title("🏠 114年度全國房地產交易分析")
st.caption("資料來源:內政部不動產交易實價查詢服務網（實價登錄）")

DB_PATH = "Database/全國房屋實價登錄資料.db"

# 資料表命名格式:S1_台北市房地產交易資料_不含車位(中古屋)
TABLE_PATTERN = re.compile(
    r"^(S\d)_(.+?)房地產交易資料_(不含車位|含車位)\((中古屋|預售屋)\)$"
)


# ------------------------------------------------------------
# 資料讀取（讀取 SQLite 資料庫裡所有符合命名規則的資料表）
# ------------------------------------------------------------
@st.cache_data
def load_all_data():
    """
    連線到全國房屋實價登錄資料.db,讀取所有資料表,
    並從表名解析出「季度」「縣市」「含車位/不含車位」「中古屋/預售屋」欄位。
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    df_list = []
    for t in tables:
        match = TABLE_PATTERN.match(t)
        if not match:
            continue
        season, county, car_label, house_type = match.groups()

        df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
        df["季度"] = season
        df["縣市"] = county
        df["含車位"] = (car_label == "含車位")
        df["房屋類型"] = house_type
        df_list.append(df)

    conn.close()

    if not df_list:
        return pd.DataFrame()

    result = pd.concat(df_list, ignore_index=True)
    # 有些行政區名稱會橫跨多個縣市（例如台北市、基隆市都有中正區）,
    # 用「縣市+鄉鎮市區」組成不會混淆的完整行政區名稱,供圖表分組使用
    result["行政區"] = result["縣市"] + result["鄉鎮市區"]

    # 重複值很多的文字欄位轉成 category 型態，減少記憶體用量
    for col in ["縣市", "鄉鎮市區", "行政區", "房屋類型", "建物型態", "季度", "交易標的", "主要用途"]:
                if col in result.columns:
                    result[col] = result[col].astype("category")

    return result


def extract_road_name(address):
    """
    只保留門牌地址中的「路名/段」,把巷、弄、號、樓、之X 都去掉。
    例如:臺北市中山區中山北路二段１３７巷３４號十樓之１ -> 臺北市中山區中山北路二段
    """
    if pd.isna(address):
        return address
    text = str(address)

    match = re.match(r"^(.*?(?:路|街|大道)(?:[一二三四五六七八九十]+段)?)", text)
    if match:
        return match.group(1)

    match = re.match(r"^(.*?(?:巷|弄))", text)
    if match:
        return match.group(1)

    return text


def format_roc_date(value):
    """
    把民國年格式的日期字串（例如 0900727 或 1130319）轉成可讀格式。
    從右邊切，避免年份是 2 碼或 3 碼長度不一造成誤判
    （例如 0900727 -> 民國90年07月27日，1130319 -> 民國113年03月19日）。
    """
    if pd.isna(value):
        return value
    s = str(value).strip()
    if len(s) < 5:
        return s
    year_part = s[:-4]
    month = s[-4:-2]
    day = s[-2:]
    try:
        return f"民國{int(year_part)}年{month}月{day}日"
    except ValueError:
        return s


def strip_parentheses(text):
    """
    移除文字中的括號及括號內的內容（同時處理全形／半形括號）。
    例如:住宅大樓(11層含以上有電梯) -> 住宅大樓
    """
    if pd.isna(text):
        return text
    s = str(text)
    s = re.sub(r"[（(].*?[）)]", "", s)
    return s.strip()


data = load_all_data()

def cn2int(cn):
    """樓層數字轉阿拉伯數字（同時處理中文數字文字與純數字字串）"""
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

data["移轉樓層"] = data["移轉層次"].apply(cn2int)
data["總樓層數_num"] = data["總樓層數"].apply(cn2int)
data["樓層比例"] = data["移轉樓層"] / data["總樓層數_num"]
data.loc[data["樓層比例"] > 1, "樓層比例"] = np.nan

if data.empty:
    st.warning("找不到資料庫檔案,請確認 Database 資料夾（內含 全國房屋實價登錄資料.db）是否與 dashboard.py 放在同一層。")
    st.stop()

SEASON_ORDER = ["S1", "S2", "S3", "S4"]


# ------------------------------------------------------------
# 側邊欄篩選條件
# ------------------------------------------------------------
st.sidebar.header("篩選條件")

seasons = sorted(data["季度"].dropna().unique())
selected_seasons = st.sidebar.multiselect(
    "選擇季度", seasons, default=seasons
)

# 縣市篩選（預設只選台北市,避免一次載入全國 368 個行政區造成圖表過於雜亂）
counties = sorted(data["縣市"].dropna().unique())
default_county = ["台北市"] if "台北市" in counties else counties[:1]
selected_counties = st.sidebar.multiselect(
    "選擇縣市", counties, default=default_county
)

# 行政區選項只列出「目前選擇的縣市」底下的行政區,避免不同縣市同名行政區混淆
county_scoped = data[data["縣市"].isin(selected_counties)]
districts = sorted(county_scoped["行政區"].dropna().unique())
district_labels = {d: d for d in districts}  # 顯示用（已含縣市，不需再轉換）

select_all_districts = st.sidebar.checkbox("全選行政區", value=True)
if select_all_districts:
    selected_districts = districts
else:
    selected_districts = st.sidebar.multiselect("選擇行政區", districts)

house_type_option = st.sidebar.radio(
    "房屋類型", ["全部", "只看中古屋", "只看預售屋"]
)
if house_type_option == "只看中古屋":
    house_types = ["中古屋"]
elif house_type_option == "只看預售屋":
    house_types = ["預售屋"]
else:
    house_types = ["中古屋", "預售屋"]

car_option = st.sidebar.radio(
    "車位篩選", ["全部", "只看含車位", "只看不含車位"]
)

filtered = data[
    data["縣市"].isin(selected_counties)
    & data["行政區"].isin(selected_districts)
    & data["房屋類型"].isin(house_types)
    & data["季度"].isin(selected_seasons)
]

if car_option == "只看含車位":
    filtered = filtered[filtered["含車位"]]
elif car_option == "只看不含車位":
    filtered = filtered[~filtered["含車位"]]


# ------------------------------------------------------------
# 交易筆數佔比圓餅圖
# 選擇「單一縣市」時 -> 自動切換成該縣市的行政區佔比
# 選擇「多個縣市」或「全部縣市」時 -> 顯示全國各縣市佔比
# 不受「行政區」篩選影響,只受季度/房屋類型/車位篩選影響
# ------------------------------------------------------------
pie_source = data[
    data["季度"].isin(selected_seasons)
    & data["房屋類型"].isin(house_types)
]
if car_option == "只看含車位":
    pie_source = pie_source[pie_source["含車位"]]
elif car_option == "只看不含車位":
    pie_source = pie_source[~pie_source["含車位"]]

if len(selected_counties) == 1:
    # 只選了一個縣市 -> 顯示該縣市底下的行政區佔比
    target_county = selected_counties[0]
    pie_source = pie_source[pie_source["縣市"] == target_county]
    group_field = "鄉鎮市區"
    group_title = "行政區"
    chart_title = f"{target_county}各行政區交易筆數佔比"
else:
    # 選了多個縣市（或尚未選擇）-> 顯示全國各縣市佔比
    group_field = "縣市"
    group_title = "縣市"
    chart_title = "各縣市交易筆數佔比（全國）"

st.subheader(chart_title)

pie_counts = (
    pie_source.groupby(group_field).size().reset_index(name="交易筆數")
)
total_count = pie_counts["交易筆數"].sum()
pie_counts["百分比標籤"] = (
    (pie_counts["交易筆數"] / total_count * 100).round(1).astype(str) + "%"
)

pie_base = alt.Chart(pie_counts).encode(
    theta=alt.Theta("交易筆數:Q", stack=True),
    color=alt.Color(f"{group_field}:N", title=group_title),
    tooltip=[
        alt.Tooltip(f"{group_field}:N", title=group_title),
        alt.Tooltip("交易筆數:Q", title="交易筆數"),
    ],
)

pie_chart = pie_base.mark_arc(outerRadius=140)
pie_labels = pie_base.mark_text(radius=160, size=11).encode(
    text=alt.Text("百分比標籤:N")
)

st.altair_chart(pie_chart + pie_labels, use_container_width=True)


# ------------------------------------------------------------
# 關鍵指標（KPI）
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("交易筆數", f"{len(filtered):,}")
col2.metric("平均每坪單價（萬元）", f"{filtered['單價_萬元每坪'].mean():.1f}")
col3.metric("平均總坪數", f"{filtered['總坪數'].mean():.1f}")


# ------------------------------------------------------------
# 各行政區平均單價比較
# ------------------------------------------------------------
st.subheader("各行政區平均單價（萬元/坪）")

district_avg = (
    filtered.groupby("行政區")["單價_萬元每坪"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

district_bar = (
    alt.Chart(district_avg)
    .mark_bar()
    .encode(
        x=alt.X("行政區:N", sort="-y", axis=alt.Axis(labelAngle=0), title="行政區"),
        y=alt.Y("單價_萬元每坪:Q", title="平均單價（萬元/坪）"),
        tooltip=[
            alt.Tooltip("行政區:N", title="行政區"),
            alt.Tooltip("單價_萬元每坪:Q", title="平均單價", format=".1f"),
        ],
    )
)
st.altair_chart(district_bar, use_container_width=True)


# ------------------------------------------------------------
# S1~S4 價格走勢折線圖（依行政區 x 房屋類型 分色）
# ------------------------------------------------------------
st.subheader("S1~S4 平均單價走勢（萬元/坪）")

trend_data = (
    filtered.groupby(["季度", "行政區", "房屋類型"])["單價_萬元每坪"]
    .mean()
    .reset_index()
)
trend_data["群組"] = trend_data["行政區"].astype(str) + " - " + trend_data["房屋類型"].astype(str)

trend_color = alt.Color(
    "群組:N",
    title="行政區 / 房屋類型",
    legend=alt.Legend(orient="right", symbolSize=120, labelFontSize=12, titleFontSize=13),
)

trend_line = alt.Chart(trend_data).mark_line().encode(
    x=alt.X("季度:N", sort=SEASON_ORDER, axis=alt.Axis(labelAngle=0), title="季度"),
    y=alt.Y("單價_萬元每坪:Q", title="平均單價（萬元/坪）"),
    color=trend_color,
)

trend_points = alt.Chart(trend_data).mark_point(size=100, filled=True).encode(
    x=alt.X("季度:N", sort=SEASON_ORDER),
    y="單價_萬元每坪:Q",
    color=alt.Color("群組:N", legend=None),
    tooltip=[
        alt.Tooltip("群組:N", title="行政區 / 房屋類型"),
        alt.Tooltip("季度:N", title="季度"),
        alt.Tooltip("單價_萬元每坪:Q", title="平均單價", format=".1f"),
    ],
)

trend_labels = alt.Chart(trend_data).mark_text(dy=-12, fontSize=11).encode(
    x=alt.X("季度:N", sort=SEASON_ORDER),
    y="單價_萬元每坪:Q",
    text=alt.Text("單價_萬元每坪:Q", format=".1f"),
    color=alt.Color("群組:N", legend=None),
)

st.altair_chart(trend_line + trend_points + trend_labels, use_container_width=True)
st.caption("圖例中每個顏色代表一組「行政區 - 房屋類型」,對應圖上同色的線與數據點。")


# ------------------------------------------------------------
# 中古屋 vs 預售屋 單價比較
# ------------------------------------------------------------
st.subheader("中古屋 vs 預售屋 平均單價比較")

type_avg = (
    filtered.groupby(["行政區", "房屋類型"])["單價_萬元每坪"]
    .mean()
    .reset_index()
)

type_bar = (
    alt.Chart(type_avg)
    .mark_bar()
    .encode(
        x=alt.X("行政區:N", axis=alt.Axis(labelAngle=0), title="行政區"),
        y=alt.Y("單價_萬元每坪:Q", title="平均單價（萬元/坪）"),
        color=alt.Color("房屋類型:N", title="房屋類型"),
        xOffset="房屋類型:N",
        tooltip=[
            alt.Tooltip("行政區:N", title="行政區"),
            alt.Tooltip("房屋類型:N", title="房屋類型"),
            alt.Tooltip("單價_萬元每坪:Q", title="平均單價", format=".1f"),
        ],
    )
)
st.altair_chart(type_bar, use_container_width=True)


# ------------------------------------------------------------
# 原始資料表（可自選要顯示的欄位；門牌只顯示到路名，可下載完整版）
# ------------------------------------------------------------
st.subheader("篩選後的原始資料")

# 欄位顯示用中文名稱對照（沒有列在這裡的欄位會直接用原始欄名，也不會出現在勾選選單中）
COLUMN_LABELS = {
    "鄉鎮市區": "鄉鎮市區",
    "土地位置建物門牌": "門牌（僅顯示路名）",
    "交易標的": "交易標的",
    "主要用途": "主要用途",
    "房屋類型": "房屋類型",
    "季度": "季度",
    "交易年月日": "交易日期",
    "移轉層次": "移轉層次",
    "總樓層數": "總樓層數",
    "建物型態": "建物型態",
    "建築完成年月": "建築完成日期",
    "屋齡": "屋齡（年）",
    "建物現況格局-房": "房數",
    "建物現況格局-廳": "廳數",
    "建物現況格局-衛": "衛浴數",
    "總坪數": "總坪數",
    "總價元": "總價（元）",
    "單價_萬元每坪": "單價（萬元/坪）",
}

# 預設勾選的精選欄位（只取資料裡實際存在的欄位，避免舊資料表缺欄位時報錯）
DEFAULT_COLUMNS = [
    "鄉鎮市區", "土地位置建物門牌", "房屋類型", "季度",
    "交易年月日", "建物型態", "屋齡",
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛",
    "總坪數", "單價_萬元每坪",
]

available_columns = [c for c in filtered.columns if c in COLUMN_LABELS]
default_columns = [c for c in DEFAULT_COLUMNS if c in available_columns]

selected_columns = st.multiselect(
    "選擇要顯示的欄位",
    options=available_columns,
    default=default_columns,
    format_func=lambda c: COLUMN_LABELS.get(c, c),
)

if not selected_columns:
    st.info("請至少選擇一個欄位才能顯示資料表。")
else:
    display_df = filtered[selected_columns].copy()

    if "土地位置建物門牌" in display_df.columns:
        display_df["土地位置建物門牌"] = display_df["土地位置建物門牌"].apply(extract_road_name)
    if "交易年月日" in display_df.columns:
        display_df["交易年月日"] = display_df["交易年月日"].apply(format_roc_date)
    if "建築完成年月" in display_df.columns:
        display_df["建築完成年月"] = display_df["建築完成年月"].apply(format_roc_date)
    if "建物型態" in display_df.columns:
        display_df["建物型態"] = display_df["建物型態"].apply(strip_parentheses)

    display_df = display_df.rename(columns=COLUMN_LABELS)
    st.dataframe(display_df, use_container_width=True)

csv = filtered.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="下載篩選後的完整資料 (CSV)",
    data=csv,
    file_name="篩選後房地產資料.csv",
    mime="text/csv",
)

# ------------------------------------------------------------
# 房屋估價工具
# ------------------------------------------------------------
st.header("🏷️ 房屋估價工具")

#------載入模型------
@st.cache_resource
def load_models():
    model_中古屋 = joblib.load("model_中古屋_隨機森林.pkl")
    model_預售屋 = joblib.load("model_預售屋_隨機森林.pkl")
    return model_中古屋, model_預售屋

model_中古屋, model_預售屋 = load_models()

#------模式切換------
估價模式 = st.radio("估價模式", ["快速行情查詢", "個人化估價"])
坪數鎖住 = (估價模式 == "快速行情查詢")

#------欄位類別(下拉選單)------
房屋類型_輸入 = st.selectbox("房屋類型", ["中古屋", "預售屋"])
縣市_輸入 = st.selectbox("縣市", counties)

行政區_選項 = data[data["縣市"] == 縣市_輸入]
行政區_選項 = sorted(行政區_選項["行政區"].dropna().unique())
行政區_輸入 = st.selectbox("行政區", 行政區_選項)

車位_輸入 = st.selectbox("車位", ["含車位", "不含車位"]) == "含車位"
建物型態選項 = [
    t for t in data["建物型態"].dropna().unique()
    if any(keyword in t for keyword in ["公寓", "華廈", "住宅大樓", "透天厝"])
]
建物型態_輸入 = st.selectbox("建物型態", sorted(建物型態選項))
季度_輸入 = st.selectbox("季度", SEASON_ORDER)

#------總坪數：依模式決定用區間選擇或直接輸入------
if 估價模式 == "快速行情查詢":
    坪數區間選項 = ["10坪以下", "11~20坪", "21~30坪", "31~40坪", "41~50坪", "51~60坪", "61坪以上"]
    坪數區間_輸入 = st.selectbox("總坪數範圍", 坪數區間選項)

    坪數區間對照 = {
        "10坪以下": (0, 10), "11~20坪": (11, 20), "21~30坪": (21, 30),
        "31~40坪": (31, 40), "41~50坪": (41, 50), "51~60坪": (51, 60),
        "61坪以上": (61, 9999),
    }
    坪數下限, 坪數上限 = 坪數區間對照[坪數區間_輸入]

    floor_source = data[
        (data["行政區"] == 行政區_輸入)
        & (data["房屋類型"] == 房屋類型_輸入)
        & (data["建物型態"] == 建物型態_輸入)
        & (data["總坪數"] >= 坪數下限)
        & (data["總坪數"] <= 坪數上限)
    ]

    if len(floor_source) == 0:
        st.warning("這個地區/房屋類型/坪數區間組合資料不足，無法估價。")
        st.stop()

    總坪數_輸入 = floor_source["總坪數"].median()
    坪數說明 = f"（依您選擇的「{坪數區間_輸入}」區間，系統帶入實際中位數 {總坪數_輸入:.1f} 坪）"

else:  # 個人化估價
    floor_source = data[
        (data["行政區"] == 行政區_輸入)
        & (data["房屋類型"] == 房屋類型_輸入)
        & (data["建物型態"] == 建物型態_輸入)
    ]

    if len(floor_source) == 0:
        st.warning("這個地區/房屋類型組合資料不足，無法估價。")
        st.stop()

    總坪數_輸入 = st.number_input("總坪數", min_value=1.0, value=30.0)
    坪數說明 = ""

#------其餘欄位自動帶入（跟坪數無關，兩種模式都一樣）------
房_預設 = floor_source["建物現況格局-房"].median()
廳_預設 = floor_source["建物現況格局-廳"].median()
衛_預設 = floor_source["建物現況格局-衛"].median()
移轉樓層_預設 = floor_source["移轉樓層"].median()
總樓層數_預設 = floor_source["總樓層數_num"].median()
屋齡_預設 = floor_source["屋齡"].median() if 房屋類型_輸入 == "中古屋" else None

caption_text = (
    f"系統帶入：{房_預設:.0f}房{廳_預設:.0f}廳{衛_預設:.0f}衛、"
    f"第{移轉樓層_預設:.0f}層／共{總樓層數_預設:.0f}層"
)
if 屋齡_預設 is not None:
    caption_text += f"、屋齡{屋齡_預設:.0f}年"
if 坪數說明:
    caption_text += "\n" + 坪數說明
st.caption(caption_text)

#------按下按鈕才觸發預測------
if st.button("開始估價"):
    樓層比例_輸入 = 移轉樓層_預設 / 總樓層數_預設

    input_data = {
        "行政區": [行政區_輸入],
        "含車位": [車位_輸入],
        "季度": [季度_輸入],
        "總坪數": [總坪數_輸入],
        "建物型態": [建物型態_輸入],
        "建物現況格局-房": [房_預設],
        "建物現況格局-廳": [廳_預設],
        "建物現況格局-衛": [衛_預設],
        "移轉樓層": [移轉樓層_預設],
        "總樓層數_num": [總樓層數_預設],
        "樓層比例": [樓層比例_輸入],
    }
    if 房屋類型_輸入 == "中古屋":
        input_data["屋齡"] = [屋齡_預設]

    input_df = pd.DataFrame(input_data)

    模型 = model_中古屋 if 房屋類型_輸入 == "中古屋" else model_預售屋
    預測單價 = 模型.predict(input_df)[0]

    st.success(f"預估單價：約 {預測單價:.1f} 萬元/坪")
    st.caption(f"預估總價：約 {預測單價 * 總坪數_輸入:.0f} 萬元")