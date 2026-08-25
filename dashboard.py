import re
import sqlite3
import streamlit as st
import pandas as pd
import altair as alt

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


data = load_all_data()

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
trend_data["群組"] = trend_data["行政區"] + " - " + trend_data["房屋類型"]

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
# 原始資料表（門牌只顯示到路名，可下載完整版）
# ------------------------------------------------------------
st.subheader("篩選後的原始資料")

display_df = filtered.copy()
display_df["土地位置建物門牌"] = display_df["土地位置建物門牌"].apply(extract_road_name)
st.dataframe(display_df, use_container_width=True)

csv = filtered.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="下載篩選後的資料 (CSV，含完整門牌)",
    data=csv,
    file_name="篩選後房地產資料.csv",
    mime="text/csv",
)