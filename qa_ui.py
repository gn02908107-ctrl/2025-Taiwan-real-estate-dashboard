"""
問答功能的 Streamlit 介面（dashboard 的「💬 問問看」頁籤內容）。核心邏輯與測試在 qa_assistant.py。

dashboard.py 只需要：
    from qa_ui import qa_available, render_qa_section
    ...
    if qa_available():   # 建立「問問看」頁籤
        with 問問看頁籤:
            render_qa_section()

Secrets 設定（本機 .streamlit/secrets.toml；Streamlit Cloud 在 App settings → Secrets）：
    GROQ_API_KEY     = "gsk_..."     # 必要。沒設定時整個「問問看」頁籤隱藏，其他功能不受影響
    QA_ACCESS_CODE   = "自訂通行碼"    # 預設需要通行碼才能使用。沒設通行碼、也沒開放公開時，頁籤隱藏
    QA_PUBLIC        = true          # 選用。設成 true 就不需要通行碼，對所有訪客開放
    GROQ_MODEL       = "openai/gpt-oss-120b"  # 選用。Groq 換模型時改這裡即可，不用改程式碼
    QA_SESSION_LIMIT = 5             # 選用。每個連線（session）最多可提問幾題，預設 5

保護機制（免費額度有限：實測每題約 5 千 token，Groq 免費方案一天 20 萬 token，
全部訪客加起來一天只夠問幾十題）：
1. 通行碼：預設不完全公開。
2. 每個 session 限制題數。
3. 相同問題快取重複使用（所有訪客共用），命中時不呼叫 API、也不計入題數。
4. 額度用完（429）與其他錯誤只顯示友善訊息，原始錯誤只寫進伺服器 log。
5. 金鑰沒設定時整個區塊隱藏。
"""
import hmac
import json
import logging
import threading
from collections import OrderedDict

import streamlit as st
from streamlit.errors import StreamlitAPIException

import qa_assistant as qa

logger = logging.getLogger(__name__)

SESSION_LIMIT_DEFAULT = 5
CACHE_MAX_ENTRIES = 300  # 快取最多幾筆，超過就淘汰最舊的
MAX_QUESTION_CHARS = 200  # 單題長度上限，避免有人貼一大段文字燒 token
MAX_CODE_ATTEMPTS = 5  # 同一個 session 通行碼最多可輸錯幾次

MSG_QUOTA = "今日額度已用完，請稍後再試。"
MSG_CONNECTION = "目前無法連線到問答服務，請稍後再試。"
MSG_UNAVAILABLE = "問答服務暫時無法使用，請稍後再試。"
MSG_GENERIC = "問答服務發生錯誤，請稍後再試。"


# ------------------------------------------------------------
# 設定判斷
# ------------------------------------------------------------
def _is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_public():
    return _is_true(qa.get_secret("QA_PUBLIC", False))


def _access_code():
    code = qa.get_secret("QA_ACCESS_CODE")
    return str(code) if code else None


def session_limit():
    try:
        n = int(qa.get_secret("QA_SESSION_LIMIT", SESSION_LIMIT_DEFAULT))
    except (TypeError, ValueError):
        return SESSION_LIMIT_DEFAULT
    return n if n > 0 else SESSION_LIMIT_DEFAULT


def qa_available():
    """「問問看」頁籤要不要出現：groq 套件、金鑰都要有，而且要嘛設了通行碼、要嘛明確設成公開。
    任何一項沒滿足就整個隱藏（預設不完全公開：忘了設通行碼也不會意外變成對外開放）。"""
    return qa.Groq is not None and qa.api_key_configured() and (is_public() or _access_code() is not None)


# ------------------------------------------------------------
# 快取與 client（跨 session 共用，同一個 process 內有效）
# ------------------------------------------------------------
@st.cache_resource
def _client():
    return qa.get_client()


@st.cache_resource
def _store():
    return {"lock": threading.Lock(), "data": OrderedDict()}


def _normalize(question):
    return " ".join(question.split())


def answer_question(question, history, client, store):
    """回傳 {"answer", "trace", "cached"}。相同（模型、對話歷史、問題）命中快取時不呼叫 API。
    發生例外時原樣往外丟，由呼叫端轉成友善訊息（例外不會被快取）。"""
    question = _normalize(question)
    trimmed = qa.trim_history(history)
    key = (qa.get_model(), tuple((m["role"], m["content"]) for m in trimmed), question)

    with store["lock"]:
        hit = store["data"].get(key)
        if hit is not None:
            store["data"].move_to_end(key)
            return {**hit, "cached": True}

    answer, trace = qa.ask(question, client, trimmed)  # 呼叫 API 不放在鎖裡面，避免擋住其他訪客

    result = {"answer": answer, "trace": trace}
    if answer and answer != qa.ROUND_LIMIT_MESSAGE:  # 只快取有內容的正常回答
        with store["lock"]:
            store["data"][key] = result
            store["data"].move_to_end(key)
            while len(store["data"]) > CACHE_MAX_ENTRIES:
                store["data"].popitem(last=False)
    return {**result, "cached": False}


def friendly_error(exc):
    """把例外轉成可以給使用者看的訊息；原始錯誤（含組織代號等細節）只寫進伺服器 log。"""
    logger.error("問答呼叫失敗：%s: %s", type(exc).__name__, exc)
    g = qa.groq
    if g is not None:
        if isinstance(exc, g.RateLimitError):
            return MSG_QUOTA
        if isinstance(exc, g.APIConnectionError):  # 逾時（APITimeoutError）也是它的子類別
            return MSG_CONNECTION
        if isinstance(exc, (g.AuthenticationError, g.PermissionDeniedError, g.NotFoundError)):
            return MSG_UNAVAILABLE
    return MSG_GENERIC


# ------------------------------------------------------------
# 介面
# ------------------------------------------------------------
def _rerun_section():
    """在區塊內互動（送出通行碼、送出問題）後重畫：一般情況只重跑這個 fragment。
    但 scope="fragment" 只有在「fragment 自己的重跑」裡才合法；如果這次剛好是整頁重跑
    （例如同時有人動了左側篩選），Streamlit 會拋錯，這時改成整頁重跑，而不是讓使用者看到錯誤。"""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _unlocked():
    """公開模式或已輸入正確通行碼回傳 True；否則顯示通行碼表單並回傳 False。"""
    if is_public() or st.session_state.get("qa_unlocked"):
        return True

    attempts = st.session_state.get("qa_attempts", 0)
    if attempts >= MAX_CODE_ATTEMPTS:
        st.error("通行碼輸錯次數過多，請重新整理頁面後再試。")
        return False

    with st.form("qa_gate_form"):
        code = st.text_input("通行碼", type="password")
        submitted = st.form_submit_button("解鎖")
    if submitted:
        if hmac.compare_digest(code.encode("utf-8"), _access_code().encode("utf-8")):
            st.session_state["qa_unlocked"] = True
            _rerun_section()
        st.session_state["qa_attempts"] = attempts + 1
        if attempts + 1 >= MAX_CODE_ATTEMPTS:
            _rerun_section()  # 用完次數：立刻重畫成鎖定狀態（不再顯示表單）
        st.error("通行碼不正確。")
    return False


def _render_details(trace):
    with st.expander("查詢細節（呼叫的工具、參數、原始查詢結果）"):
        for i, t in enumerate(trace, 1):
            st.markdown(f"**{i}. `{t['tool']}`**")
            st.json(t["args"])
            try:
                st.json(json.loads(t["result"]), expanded=2)
            except (TypeError, ValueError):
                st.code(str(t["result"]))


def _render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cached"):
            st.caption("（相同問題的快取結果，未消耗提問額度）")
        if message.get("trace"):
            _render_details(message["trace"])


@st.fragment  # 在這個區塊裡互動只重跑這個函式，不會連帶重畫整個 dashboard 的圖表
def render_qa_section():
    st.subheader("💬 問問看")
    st.info(
        "**這裡固定使用全部季度（S1~S4）、含車位與不含車位合併的資料，不受左側篩選條件影響**，"
        "所以數字可能與「圖表分析」頁籤上的不同。\n\n"
        "可以問：單一行政區的平均單價、S1~S4 走勢、兩個行政區比較、行政區單價排行、含車位與不含車位的價差。"
        "想估算特定房屋條件（坪數、樓層、格局、屋齡…）的價格，請改用「房屋估價工具」頁籤。"
    )
    if not _unlocked():
        return

    messages = st.session_state.setdefault("qa_messages", [])
    st.session_state.setdefault("qa_count", 0)
    limit = session_limit()

    for message in messages:
        _render_message(message)

    remaining = limit - st.session_state["qa_count"]
    if remaining <= 0:
        st.warning(f"本次連線的提問額度已用完（每次連線最多 {limit} 題）。重新整理頁面即可重新開始。")
        return
    st.caption(f"本次連線還可以提問 {remaining} 題（相同問題的快取結果不計）")

    question = st.chat_input("輸入問題，例如：台北市大安區的中古屋平均單價是多少？")
    if not question:
        return
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"問題太長了，請縮短到 {MAX_QUESTION_CHARS} 字以內。")
        return

    _render_message({"role": "user", "content": question})
    history = [{"role": m["role"], "content": m["content"]} for m in messages]  # 只送純文字，不含查詢細節
    with st.spinner("查詢中…"):
        try:
            result = answer_question(question, history, _client(), _store())
        except Exception as e:  # noqa: BLE001 - 任何錯誤都只給使用者友善訊息
            st.error(friendly_error(e))
            return  # 失敗的問題不放進歷史，也不扣題數，使用者可以直接重問

    if not result["cached"]:
        st.session_state["qa_count"] += 1
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": result["answer"],
                     "trace": result["trace"], "cached": result["cached"]})
    _rerun_section()  # 從 session_state 重畫，讓「剩餘題數」提示與對話紀錄一起更新
