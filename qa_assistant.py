"""
房地產自然語言問答（Groq tool use）核心邏輯與測試腳本。
介面（頁籤內容、通行碼、題數限制、快取）在 qa_ui.py，由 dashboard.py 的「問問看」頁籤呼叫。

流程：使用者問題（含最近幾輪對話歷史）→ Groq 模型決定呼叫哪個工具、帶什麼參數 → 本地用 pandas
對資料庫算出答案 → 把結果回傳給模型 → 模型用自然語言整理成回答。數字一律由本地工具算出，
模型只負責「選工具、填參數、講人話」，不負責算數，也不能自己編數字。

本模組刻意不共用 dashboard 的 data：問答只需要 7 個欄位，這裡自己讀一份精簡資料
（文字欄位轉 category），避免 dashboard 完整 20 幾欄資料再複製一份而重演記憶體不足的問題。
import 這個模組本身很輕量：資料與 system prompt 都是第一次提問時才建立。

四道防呆：
1. 行政區驗證：工具收到的名稱一律比對資料庫實際存在的值。找不到、或同名鄉鎮市區橫跨多個縣市
   （例如「大安區」台北市與台中市都有）時，工具回傳錯誤與候選清單，並指示模型「回問使用者」，
   不可自行猜測。
2. 樣本數：任何一項結果的交易筆數少於 MIN_SAMPLE 筆，工具結果會帶「樣本數提醒」，
   system prompt 要求模型必須在回答中註明樣本數較少、數字僅供參考。排行榜（query_ranking）
   則直接排除少於 MIN_SAMPLE 筆的行政區；排除後不足 N 名時帶「排行提醒」，要求模型說明。
3. 超出範圍：問題不屬於 5 種查詢範本時，system prompt 要求模型不呼叫工具、不憑常識回答，
   誠實告知超出範圍。
4. 特定房屋條件的估價（有講到具體坪數、樓層、格局、屋齡等）：不用查詢工具硬湊一個資料庫裡不存在的
   組合，而是提示改用「房屋估價工具」頁籤（迴歸模型預測）。查詢彙整既有統計數字（本模組）與
   預測特定條件的價格（估價工具）是兩種不同性質的功能，彼此不取代。

執行方式（需先在 .streamlit/secrets.toml 填入 GROQ_API_KEY）：
    python qa_assistant.py          # 跑全部測試題
    python qa_assistant.py 13 14    # 只跑指定題號（節省 Groq 免費額度的每日 token）
"""
import difflib
import json
import re
import sqlite3
from functools import lru_cache

import pandas as pd
import streamlit as st

# 部分網路環境（公司網路、防毒軟體的 HTTPS 檢查）使用的根憑證只存在作業系統憑證庫，
# Python 內建的憑證清單不認，會出現 CERTIFICATE_VERIFY_FAILED。truststore 可以讓 Python
# 改用作業系統的憑證庫。這是本機環境才有的問題，所以是「選用」的：沒裝 truststore
# （例如 Streamlit Cloud）或環境不支援時直接略過，程式照常執行。
# 必須在建立 Groq client 之前呼叫；本機使用請自行 pip install truststore。
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

try:
    import groq
    from groq import Groq
except ImportError:  # 沒裝 groq：問答功能整個停用（qa_available() 會回報），dashboard 其他功能照常
    groq = None
    Groq = None

DB_PATH = "Database/全國房屋實價登錄資料.db"
# 預設模型。原本指定 llama-3.3-70b-versatile，但 Groq 已將它下架（404 model_not_found），
# 改用同樣支援 tool use 的 gpt-oss-120b。以後 Groq 再換模型時，不用改程式碼：
# 在 st.secrets 設定 GROQ_MODEL 即可覆蓋（可用模型清單：Groq().models.list()）。
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 5  # 單一問題最多讓模型呼叫幾輪工具，避免無限迴圈
MIN_SAMPLE = 10  # 交易筆數少於這個值就視為樣本數較少
HISTORY_TURNS = 3  # 多輪對話只保留最近幾輪（一問一答為一輪）的純文字歷史，控制 token 用量
ROUND_LIMIT_MESSAGE = "（超過工具呼叫輪數上限）"  # ask() 因輪數上限而放棄時的回傳；UI 不快取這種結果


def get_secret(name, default=None):
    """讀 st.secrets；沒有 secrets 檔、沒有這個鍵、或值是空字串，都回傳 default（不丟例外）"""
    try:
        value = st.secrets[name]
    except Exception:
        return default
    return default if value in (None, "") else value


def get_model():
    return str(get_secret("GROQ_MODEL", DEFAULT_MODEL))

# 資料表命名格式與 dashboard.py 相同：S1_台北市房地產交易資料_不含車位(中古屋)
TABLE_PATTERN = re.compile(
    r"^(S\d)_(.+?)房地產交易資料_(不含車位|含車位)\((中古屋|預售屋)\)$"
)
HOUSE_TYPES = ["中古屋", "預售屋"]
SEASONS = ["S1", "S2", "S3", "S4"]


# ------------------------------------------------------------
# 資料層：只讀需要的兩個欄位，欄位處理方式與 dashboard.py 的 load_all_data 一致
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    frames = []
    for t in tables:
        m = TABLE_PATTERN.match(t)
        if not m:
            continue
        season, county, car_label, house_type = m.groups()
        df = pd.read_sql_query(f'SELECT 鄉鎮市區, 單價_萬元每坪 FROM "{t}"', conn)
        df["季度"] = season
        df["縣市"] = county
        df["含車位"] = (car_label == "含車位")
        df["房屋類型"] = house_type
        frames.append(df)
    conn.close()

    data = pd.concat(frames, ignore_index=True)
    # 有些行政區名稱橫跨多個縣市（例如台北市、基隆市都有中正區），用「縣市+鄉鎮市區」避免混淆
    # （字串相加要在轉成 category 之前做，category 型態不支援直接用 + 合併）
    data["行政區"] = data["縣市"] + data["鄉鎮市區"]
    # 重複值很多的文字欄位轉 category 省記憶體。之後所有 groupby 都要明確帶 observed=True，
    # 否則某些 pandas 版本會把「篩選後已經不存在」的類別也列成空的一列，行政區數量就會算錯。
    for col in ["縣市", "鄉鎮市區", "行政區", "季度", "房屋類型"]:
        data[col] = data[col].astype("category")
    return data


@lru_cache(maxsize=1)
def valid_areas():
    return frozenset(load_data()["行政區"].unique())


@lru_cache(maxsize=1)
def valid_counties():
    return sorted(load_data()["縣市"].unique())


# ------------------------------------------------------------
# 防呆 1：名稱驗證。台/臺 視為同一個字（資料庫本身就混用：台北市、臺東縣、臺東縣台東市）
# ------------------------------------------------------------
class ToolError(Exception):
    """工具執行失敗。payload 會原樣（JSON）回給模型，其中的「指示」告訴模型下一步該怎麼做。"""

    def __init__(self, payload):
        super().__init__(payload.get("error", ""))
        self.payload = payload


_ASK_USER = "請直接回問使用者，不要自行猜測，也不要改用其他名稱重試"


def _norm(s):
    return str(s).strip().replace("台", "臺")


@lru_cache(maxsize=1)
def _area_index():
    """正規化後的完整行政區名稱 → 資料庫實際寫法"""
    return {_norm(a): a for a in valid_areas()}


@lru_cache(maxsize=1)
def _district_index():
    """正規化後的鄉鎮市區名稱（不含縣市）→ 擁有這個名稱的所有完整行政區"""
    pairs = load_data()[["行政區", "鄉鎮市區"]].drop_duplicates()
    index = {}
    for full, district in zip(pairs["行政區"], pairs["鄉鎮市區"]):
        index.setdefault(_norm(district), []).append(full)
    return index


def resolve_area(area):
    """把使用者給的名稱對應到資料庫實際存在的行政區，對應不到或不唯一就拋 ToolError。"""
    key = _norm(area)
    if key in _area_index():
        return _area_index()[key]

    candidates = _district_index().get(key)  # 使用者只說了鄉鎮市區，例如「板橋區」「大安區」
    if candidates and len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ToolError({
            "error": "行政區名稱有多個候選（同名的鄉鎮市區橫跨多個縣市）",
            "輸入": area,
            "候選行政區": sorted(candidates),
            "指示": _ASK_USER + "，並列出候選讓使用者選擇是哪一個縣市。",
        })

    close = difflib.get_close_matches(key, list(_area_index()), n=5, cutoff=0.6)
    raise ToolError({
        "error": "資料庫找不到這個行政區",
        "輸入": area,
        "相近的行政區": [_area_index()[c] for c in close],
        "指示": _ASK_USER + "，可列出相近的行政區請使用者確認。",
    })


def resolve_county(county):
    if county == "全國":
        return county
    for c in valid_counties():
        if _norm(c) == _norm(county):
            return c
    raise ToolError({
        "error": "資料未涵蓋這個縣市",
        "輸入": county,
        "資料涵蓋的縣市": valid_counties(),
        "指示": "請告知使用者資料未涵蓋該縣市，不要改查其他縣市。",
    })


def _check_house_type(house_type):
    if house_type not in HOUSE_TYPES:
        raise ToolError({
            "error": "房屋類型無效",
            "輸入": house_type,
            "可用的房屋類型": HOUSE_TYPES,
            "指示": _ASK_USER + "，問使用者要查中古屋還是預售屋。",
        })


# ------------------------------------------------------------
# 防呆 2：樣本數提醒
# ------------------------------------------------------------
def _sample_note(counts):
    """counts: {項目名稱: 交易筆數}。有任何一項少於 MIN_SAMPLE 就回傳要併進結果的提醒欄位。"""
    low = {label: n for label, n in counts.items() if n < MIN_SAMPLE}
    if not low:
        return {}
    detail = "、".join(f"{label}僅 {n} 筆" for label, n in low.items())
    return {"樣本數提醒": f"{detail}（少於 {MIN_SAMPLE} 筆），樣本數較少，平均值僅供參考，回答時必須明確註明"}


def _subset(area, house_type):
    name = resolve_area(area)
    _check_house_type(house_type)
    d = load_data()
    return name, d[(d["行政區"] == name) & (d["房屋類型"] == house_type)]


def _mean(df):
    """未四捨五入的平均單價；漲跌幅、差額都要用它算，避免拿四捨五入後的數字再算一次造成誤差"""
    return float(df["單價_萬元每坪"].mean()) if len(df) else None


def _pct(new, old):
    return round((new / old - 1) * 100, 2) if (new is not None and old) else None


def _stats(df):
    m = _mean(df)
    return {
        "平均單價_萬元每坪": round(m, 2) if m is not None else None,
        "交易筆數": int(len(df)),
    }


# ------------------------------------------------------------
# 5 個工具的本地實作
# ------------------------------------------------------------
def query_single_area(area, house_type):
    """單一行政區＋房屋類型：平均單價、交易筆數（S1~S4、含/不含車位全部合併，與 dashboard 預設一致）"""
    name, df = _subset(area, house_type)
    return {"行政區": name, "房屋類型": house_type, **_stats(df),
            **_sample_note({name: len(df)})}


def query_trend(area, house_type):
    """單一行政區＋房屋類型：S1~S4 各季平均單價，以及相鄰季度、首季到末季的漲跌幅"""
    name, df = _subset(area, house_type)
    g = df.groupby("季度", observed=True)["單價_萬元每坪"].agg(["mean", "count"])
    g.index = g.index.astype(str)  # category 索引不能直接 reindex 成沒有出現過的季度
    g = g.reindex(SEASONS)
    rows, prev, first, counts = [], None, None, {}
    for season, r in g.iterrows():
        avg = None if pd.isna(r["mean"]) else float(r["mean"])  # 未四捨五入，只在輸出時才 round
        n = 0 if pd.isna(r["count"]) else int(r["count"])
        counts[season] = n
        rows.append({
            "季度": season,
            "平均單價_萬元每坪": round(avg, 2) if avg is not None else None,
            "交易筆數": n,
            "較上一季漲跌幅_百分比": _pct(avg, prev),
        })
        if avg is not None:
            first = avg if first is None else first
            prev = avg
    return {
        "行政區": name, "房屋類型": house_type,
        "各季": rows, "首季到末季整體漲跌幅_百分比": _pct(prev, first),
        **_sample_note(counts),
    }


def compare_areas(area_a, area_b, house_type):
    """比較兩個行政區（同一房屋類型）的平均單價"""
    name_a, da = _subset(area_a, house_type)
    name_b, db = _subset(area_b, house_type)
    pa, pb = _mean(da), _mean(db)
    return {
        "房屋類型": house_type,
        name_a: _stats(da),
        name_b: _stats(db),
        "單價差_A減B_萬元每坪": round(pa - pb, 2) if (pa is not None and pb is not None) else None,
        "A比B高出_百分比": _pct(pa, pb),
        **_sample_note({name_a: len(da), name_b: len(db)}),
    }


def query_ranking(county, house_type, order, top_n=5):
    """某縣市（或全國）各行政區平均單價排行；order 為「最貴」或「最便宜」。

    交易筆數少於 MIN_SAMPLE 的行政區一律排除，不會出現在排行榜（1~2 筆交易的平均值不具代表性，
    卻很容易擠進「最貴」或「最便宜」）。排除後如果剩下的行政區比要求的前 N 名少，
    結果會帶「排行提醒」，讓模型在回答裡說明，而不是靜默地回傳比預期少的名單。
    """
    county = resolve_county(county)
    _check_house_type(house_type)
    if order not in ("最貴", "最便宜"):
        raise ValueError(f"order 只能是「最貴」或「最便宜」，收到「{order}」")
    top_n = max(1, min(int(top_n), 20))  # 模型有時會把數字當字串傳進來

    d = load_data()
    d = d[d["房屋類型"] == house_type]
    if county != "全國":
        d = d[d["縣市"] == county]
    g = d.groupby("行政區", observed=True)["單價_萬元每坪"].agg(["mean", "count"])
    eligible = g[g["count"] >= MIN_SAMPLE]  # 樣本數門檻與「樣本數提醒」共用同一個 MIN_SAMPLE
    total, excluded = len(g), len(g) - len(eligible)
    ranked = eligible.sort_values("mean", ascending=(order == "最便宜")).head(top_n)

    result = {
        "範圍": county, "房屋類型": house_type, "排序": order,
        "有資料的行政區數": total,
        "排行": [
            {"名次": i + 1, "行政區": name,
             "平均單價_萬元每坪": round(float(r["mean"]), 2), "交易筆數": int(r["count"])}
            for i, (name, r) in enumerate(ranked.iterrows())
        ],
    }
    if excluded:
        result["已排除樣本數不足的行政區數"] = excluded
    if len(eligible) < top_n:
        # 排行比要求的少，有兩個彼此獨立的成因，可能只有一個、也可能同時成立，
        # 所以各自判斷、各自寫進說明，不能只挑一個講（否則會讓人誤以為不排除就湊得滿 N 名）：
        #   (a) 這個範圍本來就沒有 N 個行政區有資料（例如縣市的行政區數本來就少）
        #   (b) 有行政區因樣本數不足被排除
        # 各項寫成彼此獨立、以句號結尾的編號句，而不是用逗號串成一句：實測模型會把逗號後面接的
        # 「實際可比較的僅有 X 個」改寫成「因此…」，暗示是排除造成湊不滿 N 名。
        facts = []
        if total < top_n:
            facts.append(f"這個範圍內本來就只有 {total} 個行政區有資料（不受排除與否影響）")
        if excluded:
            facts.append(f"已排除樣本數不足（少於 {MIN_SAMPLE} 筆）的行政區 {excluded} 個")
        facts.append(f"實際可比較的僅有 {len(eligible)} 個（少於要求的前 {top_n} 名）")
        numbered = "".join(f"{'①②③'[i]}{fact}。" for i, fact in enumerate(facts))
        result["排行提醒"] = (
            numbered + "回答時請逐項獨立陳述以上各點，項與項之間不要加「因此、所以、因為」等連接詞"
        )
    return result


def query_parking_effect(area, house_type):
    """同一行政區＋房屋類型：含車位 vs 不含車位的平均單價差異"""
    name, d = _subset(area, house_type)
    dw, dn = d[d["含車位"]], d[~d["含車位"]]
    pw, pn = _mean(dw), _mean(dn)
    return {
        "行政區": name, "房屋類型": house_type,
        "含車位": _stats(dw), "不含車位": _stats(dn),
        "單價差_含車位減不含車位_萬元每坪": round(pw - pn, 2) if (pw is not None and pn is not None) else None,
        "含車位比不含車位高出_百分比": _pct(pw, pn),
        **_sample_note({"含車位": len(dw), "不含車位": len(dn)}),
    }


TOOL_FUNCS = {
    "query_single_area": query_single_area,
    "query_trend": query_trend,
    "compare_areas": compare_areas,
    "query_ranking": query_ranking,
    "query_parking_effect": query_parking_effect,
}


# ------------------------------------------------------------
# 給模型看的工具定義（OpenAI 相容格式）
# ------------------------------------------------------------
_AREA_DESC = (
    "使用者提到的行政區名稱，原樣傳入即可：可以是完整名稱（縣市+鄉鎮市區，例如「台北市大安區」），"
    "也可以只有鄉鎮市區（例如「板橋區」）。使用者沒說縣市時，不要自行補上縣市，工具會自己比對資料庫。"
)
_TYPE_PARAM = {
    "type": "string", "enum": HOUSE_TYPES,
    "description": "房屋類型：中古屋 或 預售屋",
}


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _tool(
        "query_single_area",
        "查詢單一行政區＋房屋類型的平均單價（萬元/坪）與交易筆數。",
        {"area": {"type": "string", "description": _AREA_DESC}, "house_type": _TYPE_PARAM},
        ["area", "house_type"],
    ),
    _tool(
        "query_trend",
        "查詢單一行政區＋房屋類型在 S1~S4 各季的平均單價，以及各季漲跌幅與整體漲跌幅。",
        {"area": {"type": "string", "description": _AREA_DESC}, "house_type": _TYPE_PARAM},
        ["area", "house_type"],
    ),
    _tool(
        "compare_areas",
        "比較兩個行政區（同一房屋類型）的平均單價，回傳各自均價、差額與高出的百分比。",
        {
            "area_a": {"type": "string", "description": "第一個行政區。" + _AREA_DESC},
            "area_b": {"type": "string", "description": "第二個行政區。" + _AREA_DESC},
            "house_type": _TYPE_PARAM,
        },
        ["area_a", "area_b", "house_type"],
    ),
    _tool(
        "query_ranking",
        f"查詢某縣市（或全國）各行政區的平均單價排行，可選最貴或最便宜。交易筆數少於 {MIN_SAMPLE} 筆的行政區會自動排除，"
        "不會出現在排行榜。",
        {
            "county": {
                "type": "string",
                "description": "縣市名稱；要查全國排行請填「全國」。使用者指定的縣市即使不在資料範圍內也照實填入，工具會回報。",
            },
            "house_type": _TYPE_PARAM,
            "order": {"type": "string", "enum": ["最貴", "最便宜"], "description": "排序方向"},
            "top_n": {"type": "integer", "description": "要列出前幾名，預設 5，最多 20"},
        },
        ["county", "house_type", "order"],
    ),
    _tool(
        "query_parking_effect",
        "比較同一行政區＋房屋類型下，含車位與不含車位的平均單價差異。",
        {"area": {"type": "string", "description": _AREA_DESC}, "house_type": _TYPE_PARAM},
        ["area", "house_type"],
    ),
]

@lru_cache(maxsize=1)
def build_system_prompt():
    """system prompt 含資料涵蓋的縣市清單，所以要等資料載入後才能建立（第一次提問時才建，import 時不碰資料庫）"""
    return f"""你是台灣不動產實價登錄資料（西元 2025 年＝民國 114 年，S1~S4 四季）的問答助理。
你只能回答下列 5 種查詢，其他問題一律不回答（特定房屋條件的估價請依規則 5 處理）：
  (1) 單一行政區＋房屋類型的平均單價與交易筆數
  (2) 單一行政區＋房屋類型的 S1~S4 各季走勢與漲跌幅
  (3) 兩個行政區的平均單價比較
  (4) 某縣市或全國的行政區單價排行（最貴／最便宜）
  (5) 同一行政區含車位 vs 不含車位的單價差異

規則：
1. 所有數字都必須呼叫工具查詢，不可憑空猜測或自行估算。單價單位是「萬元/坪」，房屋類型只有「中古屋」與「預售屋」。
2. 行政區參數：使用者怎麼說就怎麼傳（可以只有鄉鎮市區，例如「大安區」），絕對不要自行補上或猜測縣市，工具會自己比對資料庫。
   反過來也不可以刪減：使用者連縣市一起說了，就整段原樣傳入（說「台北市火星區」就傳「台北市火星區」，不可只傳「火星區」）。
   即使你覺得這個行政區可能不存在，也照傳，由工具判斷是否存在並提供相近的候選。
   資料涵蓋的縣市：{'、'.join(valid_counties())}。
3. 工具回傳 error 時（找不到行政區、名稱有多個候選、房屋類型無效等）：不可自行猜測、不可改用其他名稱重試，必須直接回問使用者，
   並列出工具提供的候選讓使用者選擇。使用者沒有說明房屋類型時，同樣先回問，不要擅自決定。
   使用者指定的縣市不在資料範圍內時，直接告知資料未涵蓋，不要改查其他縣市。
   對話中如果你先前回問過使用者，使用者接著的簡短回覆（例如只回「台北市」）就是在回答你的提問，
   請結合前文的問題重新查詢，不要當成新問題。
4. 工具結果含「樣本數提醒」時，回答中必須明確註明哪一項樣本數較少（少於 {MIN_SAMPLE} 筆）、數字僅供參考。
   排行結果含「排行提醒」時，回答中必須逐項說明其內容（工具已把各點編成 ①②③），不可以靜默地只列出比使用者要求更少的名單。
   請把每一點當成彼此獨立的事實並列陳述，項與項之間不要加「因此、所以、因為、由於、導致」等任何因果連接詞
   （連「因此」也不行）。因果寫法會暗示「沒有排除就湊得滿要求的名數」，但事實可能是本來就湊不滿。
   範例（數字只是示範，請換成工具實際回傳的數字）：
     - 這個範圍內本來就只有 9 個行政區有資料，不受排除與否影響。
     - 其中 2 個行政區樣本數不足（少於 {MIN_SAMPLE} 筆），已排除。
     - 實際可比較的僅有 7 個，少於您要求的前 12 名。
   工具提醒裡只有其中幾點時，照實陳述有的那幾點，不要編造沒有的。
5. 使用者問的是「特定房屋條件」的估價（例如提到具體坪數、樓層、格局、屋齡、建物型態等細節，想知道這樣一間房子大概值多少）：
   本條只在使用者描述了具體房屋條件、並問「這樣一間房子值多少」時適用；單純查詢某個行政區的平均單價、走勢、排行等，
   （包含行政區名稱打錯或不存在的情況）都屬於規則 1～3，不適用本條，照常呼叫查詢工具。
   適用本條時，不要呼叫查詢工具，也不要自己估價。請告知這類問題要改用頁面上的「房屋估價工具」頁籤
   （它用迴歸模型針對指定條件預測），並說明這裡只能查詢彙整過的歷史統計數字（上述 5 種查詢）。
   （以下語氣要求只適用於這種估價問題，其他情況不要套用。）語氣要像「主動把使用者引導去更適合的工具」：
   先告訴使用者「房屋估價工具」頁籤更適合這個問題，它能依使用者提供的坪數、樓層、屋齡等條件直接預測；
   不要說「依照規則／規範／系統設定所以不能回答」，也不要用道歉或拒絕的口氣。
   引導之後不需要把 5 種查詢完整列出，最多用一句話帶過這裡能查的歷史統計。
6. 問題不屬於上述 5 種查詢、也不是規則 5 的估價時（例如預測未來房價走勢、買房或投資建議、貸款、與房價無關的問題）：
   不要呼叫工具，也不要憑常識回答。拒絕時，第一句就要明確點出使用者要的是什麼，並說明這件事這裡做不到
   （例如「您想請我寫一首關於房子的詩，這不在我能協助的範圍內」、「預測明年房價走勢與買房建議，是這裡做不到的」），
   不要只籠統地說「我只能提供某某查詢」而沒提到使用者要的事。之後再簡短列出你能回答的 5 種查詢。
7. 用繁體中文簡潔回答，並引用工具回傳的實際數字。"""


# ------------------------------------------------------------
# 問答主流程：模型選工具 → 本地執行 → 結果回傳給模型 → 直到模型不再呼叫工具
# ------------------------------------------------------------
def api_key_configured():
    key = get_secret("GROQ_API_KEY")
    return bool(key) and not str(key).startswith("在這裡")  # 「在這裡…」是 secrets.toml 範本的佔位字串


def get_client():
    if Groq is None:
        raise RuntimeError("沒有安裝 groq 套件（pip install groq）")
    if not api_key_configured():
        raise RuntimeError("請先在 .streamlit/secrets.toml 填入你的 GROQ_API_KEY")
    return Groq(api_key=get_secret("GROQ_API_KEY"))


def trim_history(history):
    """多輪對話歷史：只保留最近 HISTORY_TURNS 輪的「純文字」user/assistant 訊息，
    不含工具呼叫紀錄（後續問題需要的話，模型會重新呼叫工具），並確保從 user 訊息開始。"""
    msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    msgs = msgs[-HISTORY_TURNS * 2:]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def ask(question, client, history=None):
    """回傳 (最終回答文字, 工具呼叫紀錄 list)。
    history：先前幾輪的 [{"role": "user"/"assistant", "content": 文字}, ...]，讓「回問使用者」之後
    使用者的簡短回覆（例如只回「台北市」）能接得上前文。"""
    messages = (
        [{"role": "system", "content": build_system_prompt()}]
        + trim_history(history)
        + [{"role": "user", "content": question}]
    )
    trace = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=get_model(), messages=messages, tools=TOOLS, tool_choice="auto", temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content, trace

        messages.append(message)  # 帶有 tool_calls 的 assistant 訊息要原樣放回去
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            try:
                content = json.dumps(TOOL_FUNCS[call.function.name](**args), ensure_ascii=False)
            except ToolError as e:  # 驗證失敗：把錯誤與「指示」回給模型，讓它去回問使用者
                content = json.dumps({"error": e.payload.pop("error"), **e.payload}, ensure_ascii=False)
            except Exception as e:  # 其他錯誤（參數缺漏、型別錯誤等）也回給模型，不讓程式中斷
                content = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            trace.append({"tool": call.function.name, "args": args, "result": content})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return ROUND_LIMIT_MESSAGE, trace


# ------------------------------------------------------------
# 測試問題。每題有一個檢查函式 check(answer, trace) -> (是否通過, 原因)
# 前 5 題：5 種範本各一題，檢查模型是否選對工具、填對參數
# 後面：三道防呆各自的情境
# ------------------------------------------------------------
def _ok_calls(trace):
    return [t for t in trace if '"error"' not in t["result"]]


def expect_call(tool, args):
    """模型可能只傳「板橋區」而不是「新北市板橋區」（工具會自己對應），所以參數比對時，
    除了原始參數相等，也接受「預期的完整名稱出現在工具結果裡」（結果裡是對應後的行政區）。"""
    def check(answer, trace):
        hit = any(
            t["tool"] == tool
            and all(str(t["args"].get(k)) == str(v) or (isinstance(v, str) and v in t["result"])
                    for k, v in args.items())
            for t in _ok_calls(trace)
        )
        return hit, f"預期成功呼叫 {tool}，參數 {args}"
    return check


_ASK_WORDS = ("？", "?", "請問", "請選擇", "請您選擇", "請回覆", "請告訴", "請確認", "請提供")


def expect_ask_back(*must_mention):
    """核心：不可有任何成功的工具查詢（有就代表模型猜了一個答案）。
    另外回答要有請使用者回覆／選擇的用語，並提到指定字詞（例如候選的縣市）。"""
    def check(answer, trace):
        if _ok_calls(trace):
            return False, "模型在名稱不明確時仍查出了結果（猜測）：" + str(_ok_calls(trace)[0]["args"])
        if not any(w in answer for w in _ASK_WORDS):
            return False, "回答裡沒有請使用者回覆或選擇"
        missing = [w for w in must_mention if w not in answer]
        return (not missing), (f"回答缺少：{missing}" if missing else "已回問使用者，沒有猜測")
    return check


def expect_not_covered(word):
    def check(answer, trace):
        if _ok_calls(trace):
            return False, "資料未涵蓋的縣市卻查出了結果：" + str(_ok_calls(trace)[0]["args"])
        said = word in answer and any(w in answer for w in ("未涵蓋", "沒有", "無", "不在", "查無"))
        return said, "已告知資料未涵蓋" if said else "回答沒有明確說明資料未涵蓋"
    return check


def expect_low_sample(tool):
    def check(answer, trace):
        calls = [t for t in _ok_calls(trace) if t["tool"] == tool]
        if not calls:
            return False, f"沒有成功呼叫 {tool}"
        if "樣本數提醒" not in calls[0]["result"]:
            return False, "工具結果沒有帶樣本數提醒"
        said = "樣本" in answer and any(w in answer for w in ("少", "不足", "僅供參考"))
        return said, "回答已註明樣本數較少" if said else "工具有提醒，但回答沒有註明樣本數較少"
    return check


# 用來串起「排除」與「湊不滿 N 名」的因果連接詞。（「3 個因樣本數不足被排除」這種說明『排除原因』的
# 單字「因」不在檢查範圍，只擋明確把兩件事連成因果的詞。）
CAUSAL_WORDS = ("因為", "所以", "由於", "導致", "因此", "因而", "造成", "使得")


def _has_any(text, options):
    """options 裡每一項是字串，或「任一個都算」的字串 tuple；回傳缺少的項目"""
    return [o for o in options if not any(w in text for w in ((o,) if isinstance(o, str) else o))]


def expect_ranking(n_listed, note, answer_has=(), answer_lacks=()):
    """排行榜檢查：實際列出 n_listed 名、每一名的交易筆數都不少於 MIN_SAMPLE；
    note=True 代表工具結果必須帶「排行提醒」，False 代表不該帶；再檢查回答內容。"""
    def check(answer, trace):
        calls = [t for t in _ok_calls(trace) if t["tool"] == "query_ranking"]
        if not calls:
            return False, "沒有成功呼叫 query_ranking"
        result = json.loads(calls[0]["result"])
        entries = result["排行"]
        if len(entries) != n_listed:
            return False, f"排行應列出 {n_listed} 名，實際 {len(entries)} 名"
        low = [e["行政區"] for e in entries if e["交易筆數"] < MIN_SAMPLE]
        if low:
            return False, f"排行榜出現樣本數不足的行政區：{low}"
        if ("排行提醒" in result) != note:
            return False, f"工具結果{'應' if note else '不應'}帶「排行提醒」"
        missing = _has_any(answer, answer_has)
        if missing:
            return False, f"回答缺少：{missing}"
        present = [w for w in answer_lacks if w in answer]
        if present:
            return False, f"回答不該出現：{present}"
        return True, "排行榜已排除樣本數不足的行政區，回答說明正確"
    return check


def expect_redirect_to_estimator():
    """特定房屋條件的估價：不可呼叫查詢工具硬湊（有就代表模型拿平均單價當估價），
    回答要提示改用「房屋估價工具」"""
    def check(answer, trace):
        if trace:
            return False, "特定條件的估價問題卻呼叫了查詢工具：" + trace[0]["tool"]
        said = "估價工具" in answer
        if not said:
            return False, "回答沒有提示改用「房屋估價工具」"
        # 語氣：應是主動引導，不是「因為規則所以不能做」（關鍵字只能抓到明顯的規則口吻，完整語氣要人工讀回答）
        stiff = [w for w in ("規範", "規則", "規定", "系統設定", "抱歉", "無法直接") if w in answer]
        if stiff:
            return False, "已提示估價工具，但語氣像在拒絕／引用規則：" + "、".join(stiff)
        return True, "已提示改用房屋估價工具（語氣為引導）"
    return check


def expect_out_of_scope():
    def check(answer, trace):
        if trace:
            return False, "超出範圍的問題卻呼叫了工具：" + trace[0]["tool"]
        said = any(w in answer for w in ("無法", "超出", "範圍", "不在", "只能回答"))
        return said, "已告知超出範圍" if said else "回答沒有明確說明超出範圍"
    return check


TEST_CASES = [
    # --- 基本流程：5 種範本 ---
    ("台北市大安區的中古屋平均單價是多少？總共有幾筆交易？",
     expect_call("query_single_area", {"area": "台北市大安區", "house_type": "中古屋"})),
    ("新北市板橋區的預售屋，S1 到 S4 的價格走勢如何？漲了還是跌了？",
     expect_call("query_trend", {"area": "新北市板橋區", "house_type": "預售屋"})),
    ("台北市信義區跟台北市大安區的中古屋，哪一區平均單價比較高？差多少？",
     expect_call("compare_areas", {"area_a": "台北市信義區", "area_b": "台北市大安區", "house_type": "中古屋"})),
    ("高雄市預售屋單價最貴的前三個行政區是哪些？",
     expect_call("query_ranking", {"county": "高雄市", "house_type": "預售屋", "order": "最貴", "top_n": 3})),
    ("台中市西屯區的中古屋，含車位跟不含車位的單價差多少？",
     expect_call("query_parking_effect", {"area": "台中市西屯區", "house_type": "中古屋"})),
    # --- 防呆 1：行政區抓不到 / 不明確 → 回問，不亂猜 ---
    ("台北市火星區的中古屋平均單價是多少？", expect_ask_back()),
    ("大安區的中古屋平均單價是多少？", expect_ask_back("台北市", "台中市")),  # 台北市、台中市都有大安區
    ("連江縣的預售屋，單價最貴的行政區是哪幾個？", expect_not_covered("連江")),
    # --- 防呆 2：樣本數少於門檻 → 回答要註明 ---
    ("澎湖縣馬公市的預售屋平均單價是多少？",  # 這個組合資料庫裡只有 1 筆
     expect_low_sample("query_single_area")),
    # --- 防呆 2 補充：排行榜排除樣本數不足的行政區，剩下不夠 N 名時要在回答裡說明 ---
    # 基隆市預售屋：7 個行政區有資料，只有 4 個樣本足夠（排除 3 個）
    ("基隆市預售屋單價最貴的前五個行政區是哪些？",
     expect_ranking(4, note=True, answer_has=(("4", "四"), "排除"))),
    # 澎湖縣預售屋：唯一有資料的行政區只有 1 筆，排除後一個都不剩
    ("澎湖縣預售屋單價最貴的前三個行政區是哪些？",
     expect_ranking(0, note=True, answer_has=(("排除", "不足"),))),
    # 花蓮縣預售屋：本來就只有 4 個行政區、沒有任何一個被排除，回答不能說「已排除」
    ("花蓮縣預售屋單價最便宜的前五名是哪些行政區？",
     expect_ranking(4, note=True, answer_has=(("4", "四"),), answer_lacks=("已排除",))),
    # 兩個成因同時成立：本來就湊不滿 N 個，其中又有行政區因樣本不足被排除。
    # 回答必須把兩件事「並列」為獨立事實：要有「本來」這個獨立事實的說法，並且不能用因果連接詞串起來。
    # 基隆市預售屋要前十名：7 個行政區有資料，排除 3 個，剩 4 個
    ("基隆市預售屋單價最貴的前十個行政區是哪些？",
     expect_ranking(4, note=True, answer_has=(("4", "四"), "排除", ("7", "七"), "本來"),
                    answer_lacks=CAUSAL_WORDS)),
    # 嘉義縣預售屋要前二十名（刻意換一組跟 system prompt 範例、上一題都不同的數字）：12 個有資料，排除 4 個，剩 8 個
    ("嘉義縣預售屋單價最貴的前二十個行政區是哪些？",
     expect_ranking(8, note=True, answer_has=(("12", "十二"), ("8", "八"), "排除", "本來"),
                    answer_lacks=CAUSAL_WORDS)),
    # 全國預售屋最便宜：排除前前 5 名裡有 3 個只有 1、6、9 筆交易的行政區，現在不該再出現
    ("全國預售屋單價最便宜的前五個行政區是哪些？",
     expect_ranking(5, note=False, answer_lacks=("彰化縣芳苑鄉", "雲林縣褒忠鄉", "台南市楠西區"))),
    # --- 防呆 3：不屬於 5 種範本 → 誠實告知超出範圍 ---
    ("你覺得明年台北市大安區的房價會漲還是跌？現在適合買房嗎？", expect_out_of_scope()),
    ("可以幫我寫一首關於房子的詩嗎？", expect_out_of_scope()),
    # --- 防呆 4：特定房屋條件的估價 → 不硬答，提示改用「房屋估價工具」頁籤 ---
    ("我想買台北市大安區的中古屋，30 坪、10 樓、3 房 2 廳 2 衛、屋齡 20 年，這樣一間大概值多少錢？",
     expect_redirect_to_estimator()),
    # --- 多輪對話：使用者對「回問」的簡短回覆，要能接上前文（第三個元素是先前的對話歷史）---
    ("台北市",
     expect_call("query_single_area", {"area": "台北市大安區", "house_type": "中古屋"}),
     [{"role": "user", "content": "大安區的中古屋平均單價是多少？"},
      {"role": "assistant", "content": "「大安區」在資料庫中有兩個可能的行政區，請問您要查詢的是哪一個？\n"
                                       "- 台中市大安區\n- 台北市大安區"}]),
]


def run_tests(only=None):
    """only：只跑指定題號（1 起算）的集合；None 代表全部。
    Groq 免費額度有每日 token 上限，完整跑一輪會用掉不少，改 prompt 後只想驗證特定題目時可以只跑那幾題。"""
    client = get_client()
    cases = [(i, c) for i, c in enumerate(TEST_CASES, 1) if only is None or i in only]
    passed = 0
    for i, case in cases:
        question, check = case[0], case[1]
        history = case[2] if len(case) > 2 else None
        print(f"\n===== 測試 {i}：{question}" + ("（多輪：帶入前文）" if history else ""))
        try:
            answer, trace = ask(question, client, history)
        except Exception as e:
            print(f"[FAIL] 呼叫過程出錯：{type(e).__name__}: {e}")
            continue

        for t in trace:
            print(f"  工具：{t['tool']}  參數：{t['args']}")
            print(f"  結果：{t['result']}")
        print(f"  回答：{answer}")

        ok, reason = check(answer, trace)
        print(f"[{'OK' if ok else 'FAIL'}] {reason}")
        passed += ok
    print(f"\n通過 {passed} / {len(cases)}")


if __name__ == "__main__":
    import sys

    # python qa_assistant.py          → 跑全部題目
    # python qa_assistant.py 13 14    → 只跑第 13、14 題
    run_tests({int(a) for a in sys.argv[1:]} or None)
