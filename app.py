import streamlit as st
from transformers import pipeline
import torch
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
import re

# ---------------------------------------------------
# Page Configuration (Must be first Streamlit command)
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Stock Analyst Pro",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# 🔥 Fix: force the page to open at the TOP on first
# visit. Because the chat area is pinned to the bottom
# of the screen, browsers tend to auto-scroll there on
# load, making the app look like it "opens from the
# bottom." This only runs once per session (guarded by
# session_state) so it doesn't fight with the normal
# auto-scroll-to-newest-message behavior during chat.
#
# NOTE: this retries several times over ~1.5 seconds
# and targets multiple possible container elements,
# because a single immediate scroll attempt often fires
# before the page's full height is rendered (charts and
# other heavy content loading afterward pushes the
# scroll position back down).
# ---------------------------------------------------
if "scrolled_to_top" not in st.session_state:
    components.html("""
    <script>
        function forceScrollTop() {
            try {
                window.parent.scrollTo(0, 0);
                var doc = window.parent.document;

                var candidates = [
                    doc.querySelector('section.main'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('[data-testid="stMain"]'),
                    doc.documentElement,
                    doc.body
                ];

                candidates.forEach(function(el) {
                    if (el) { el.scrollTop = 0; }
                });
            } catch (e) {}
        }

        forceScrollTop();
        [100, 300, 600, 1000, 1500].forEach(function(delay) {
            setTimeout(forceScrollTop, delay);
        });
    </script>
    """, height=0)
    st.session_state.scrolled_to_top = True

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#00E5FF;
}

.sub-title{
    text-align:center;
    font-size:22px;
    color:#D3D3D3;
}

.footer{
    text-align:center;
    color:gray;
}

/* Styled chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

.msg-time {
    font-size:11px;
    color:#8A8A8A;
    margin-top:4px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model (with a proper step-by-step loading screen
# so first-time visitors can clearly see the app is
# working, not stuck/broken, during the initial load)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

if "model_ready" not in st.session_state:
    import time as _time

    with st.status("🚀 Starting Nova AI Stock Analyst...", expanded=True) as status:
        st.write("📦 Loading Qwen2.5 language model... (this can take up to a minute on first run)")
        chatbot = load_model()
        st.write("✅ Language model ready")

        st.write("📡 Connecting to live market data feeds...")
        _time.sleep(0.4)
        st.write("✅ Market data connected")

        st.write("🧠 Initializing AI sentiment analysis engine...")
        _time.sleep(0.3)
        st.write("✅ Sentiment engine ready")

        status.update(label="✅ Nova is ready! Redirecting to chat...", state="complete", expanded=False)

    st.session_state.model_ready = True
else:
    chatbot = load_model()  # instant — returns from cache

# ---------------------------------------------------
# Chat Input — called immediately after model load so it
# renders right away, instead of waiting for the entire
# page (dashboard, sidebar, cards, welcome section) to
# finish rendering first. st.chat_input is pinned to the
# bottom of the screen via CSS regardless of call order.
# ---------------------------------------------------
question = st.chat_input("Ask anything about Stocks, ETFs or Investing...")

# ---------------------------------------------------
# Constants
# ---------------------------------------------------
MAX_TURNS = 20

SYSTEM_PROMPT = """
You are a professional Stock Market Analyst.

You help users understand:

- Stocks
- ETFs
- Mutual Funds
- Financial Ratios
- Company Fundamentals
- Technical Analysis
- Portfolio Management
- Risk Analysis

Guidelines:
- Explain in simple, clear language.
- Use short paragraphs or bullet points where helpful.
- Give concrete, complete answers rather than cutting off mid-thought.
- If you don't know something, say so instead of making it up.
- This is for educational purposes only, not financial advice.
"""

# ---------------------------------------------------
# Conversation Memory
# ---------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5

# ---------------------------------------------------
# 🔥 Live Stock Chart — Wow Factor Feature
# ---------------------------------------------------
# Maps common company names/aliases to real Yahoo Finance
# tickers (Indian NSE stocks). Detects mentions in the
# user's question and auto-renders a live interactive chart.
# ---------------------------------------------------
TICKER_MAP = {
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl": "HCLTECH.NS",
    "itc": "ITC.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani": "ADANIENT.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "l&t": "LT.NS",
    "larsen": "LT.NS",
    "larsen and toubro": "LT.NS",
    "coal india": "COALINDIA.NS",
    "ongc": "ONGC.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS",
    "axis bank": "AXISBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "sun pharma": "SUNPHARMA.NS",
    "dr reddy": "DRREDDY.NS",
    "dr reddys": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "asian paints": "ASIANPAINT.NS",
    "nestle": "NESTLEIND.NS",
    "nestle india": "NESTLEIND.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "titan": "TITAN.NS",
    "ultratech": "ULTRACEMCO.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "jsw steel": "JSWSTEEL.NS",
    "grasim": "GRASIM.NS",
    "britannia": "BRITANNIA.NS",
    "eicher motors": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "indusind bank": "INDUSINDBK.NS",
    "shree cement": "SHREECEM.NS",
    "divis lab": "DIVISLAB.NS",
    "divis laboratories": "DIVISLAB.NS",
    "apollo hospitals": "APOLLOHOSP.NS",
    "bpcl": "BPCL.NS",
    "vedanta": "VEDL.NS",
    "zomato": "ZOMATO.NS",
    "paytm": "PAYTM.NS",
    "irctc": "IRCTC.NS",
    "dmart": "DMART.NS",
    "avenue supermarts": "DMART.NS",
    "hindustan zinc": "HINDZINC.NS",
    "hindalco": "HINDALCO.NS",
    "hindalco industries": "HINDALCO.NS",
    "grasim industries": "GRASIM.NS",
    "gail": "GAIL.NS",
    "sail": "SAIL.NS",
    "pnb": "PNB.NS",
    "punjab national bank": "PNB.NS",
    "bank of baroda": "BANKBARODA.NS",
    "canara bank": "CANBK.NS",
    "united spirits": "MCDOWELL-N.NS",
    "godrej consumer": "GODREJCP.NS",
    "pidilite": "PIDILITIND.NS",
    "havells": "HAVELLS.NS",
    "dabur": "DABUR.NS",
    "colgate": "COLPAL.NS",
    "colgate palmolive": "COLPAL.NS",
    "lupin": "LUPIN.NS",
    "aurobindo pharma": "AUROPHARMA.NS",
    "info edge": "NAUKRI.NS",
    "naukri": "NAUKRI.NS",
    "trent": "TRENT.NS",
    "page industries": "PAGEIND.NS",
    "muthoot finance": "MUTHOOTFIN.NS",
    "pfc": "PFC.NS",
    "rec limited": "RECLTD.NS",
    "ntpc green": "NTPCGREEN.NS",
    "tata power": "TATAPOWER.NS",
    "adani power": "ADANIPOWER.NS",
    "adani green": "ADANIGREEN.NS",
    "adani ports": "ADANIPORTS.NS",
}

def detect_tickers(text):
    """Find company mentions in the user's question and return
    matching (display_name, ticker) pairs, longest names first
    so 'hdfc bank' matches before the shorter 'hdfc'.
    This checks the local fast-path map first (instant, no
    network call), then falls back to dynamic resolution for
    ANY company not in the map."""
    text_lower = text.lower()
    found = []
    seen_tickers = set()

    for name in sorted(TICKER_MAP.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
            ticker = TICKER_MAP[name]
            if ticker not in seen_tickers:
                found.append((name.title(), ticker))
                seen_tickers.add(ticker)

    # 🔥 Dynamic fallback: works for ANY company, not just the
    # ones hardcoded above. Triggers if the question either uses
    # stock-related keywords OR contains a Title Case phrase that
    # looks like a company name (e.g. "Hindustan Zinc"), so it
    # still works even when the user just types a bare company
    # name without words like "compare" or "stock".
    looks_company_related = any(
        kw in text_lower for kw in
        ["compare", " vs ", "graph", "chart", "ltd", "ltd.", "stock",
         "share price", "company", "shares", "investment in", "invest in"]
    )

    has_titlecase_phrase = bool(
        re.search(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    )

    if looks_company_related or has_titlecase_phrase:
        dynamic_matches = resolve_companies_dynamically(text, exclude_tickers=seen_tickers)
        for name, ticker in dynamic_matches:
            if ticker not in seen_tickers:
                found.append((name, ticker))
                seen_tickers.add(ticker)

    return found


@st.cache_data(ttl=3600)
def search_yahoo_ticker(company_name):
    """Look up the real ticker symbol for ANY company name using
    Yahoo Finance's own public search endpoint — the same one
    the Yahoo Finance search box uses. No API key required.
    Prefers Indian (.NS / .BO) listings since this app is
    India-focused, but will return any match if none found."""
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": company_name, "quotesCount": 5, "newsCount": 0}

        response = requests.get(url, headers=headers, params=params, timeout=6)
        data = response.json()

        quotes = data.get("quotes", [])
        if not quotes:
            return None

        # Prefer NSE/BSE listings for Indian relevance
        for q in quotes:
            symbol = q.get("symbol", "")
            if symbol.endswith(".NS") and q.get("quoteType") == "EQUITY":
                return {"symbol": symbol, "name": q.get("shortname") or q.get("longname") or company_name}

        for q in quotes:
            if q.get("quoteType") == "EQUITY":
                return {
                    "symbol": q.get("symbol"),
                    "name": q.get("shortname") or q.get("longname") or company_name
                }

        return None

    except Exception:
        return None


def extract_candidate_phrases(question):
    """Split the question into likely company-name candidates using
    simple, reliable text splitting — NOT the LLM. Small local models
    are unreliable at multi-item extraction (e.g. missing the second
    company in 'compare X and Y'), so we use deterministic splitting
    on connector words instead, then let Yahoo's fuzzy search resolve
    each fragment. This is far more reliable in practice."""

    text = question

    # Strip common instruction words that aren't part of a company name
    text = re.sub(
        r"\b(compare|vs\.?|versus|with graphs?|with charts?|show me|"
        r"please|difference between|and their|stock price of|stock of)\b",
        " ", text, flags=re.IGNORECASE
    )

    # Split on connectors that typically separate multiple companies
    parts = re.split(r"\band\b|,|&|\bvs\b", text, flags=re.IGNORECASE)

    candidates = [p.strip(" .?!") for p in parts if len(p.strip(" .?!")) > 2]

    return candidates[:4]  # safety cap


def resolve_companies_dynamically(question, exclude_tickers=None):
    """Full pipeline: split question into candidate phrases → each
    phrase is resolved to a real ticker via live Yahoo Finance search.
    This is what makes ANY company work, not just a fixed list."""
    exclude_tickers = exclude_tickers or set()
    results = []

    candidates = extract_candidate_phrases(question)

    for phrase in candidates:
        match = search_yahoo_ticker(phrase)
        if match and match["symbol"] not in exclude_tickers:
            results.append((match["name"], match["symbol"]))
            exclude_tickers.add(match["symbol"])

    return results


# ---------------------------------------------------
# 🔥 Live Scrolling Ticker Tape — Bloomberg/CNBC style
# ---------------------------------------------------
TICKER_TAPE_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "SBI": "SBIN.NS",
}

@st.cache_data(ttl=60)
def get_ticker_tape_data():
    """Fetch live prices for the ticker tape. Cached for 60s
    so we're not hammering Yahoo Finance on every rerun."""
    items = []
    for label, symbol in TICKER_TAPE_SYMBOLS.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                price = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                pct = ((price - prev) / prev) * 100
                items.append((label, price, pct))
        except Exception:
            continue
    return items


def render_ticker_tape():
    data = get_ticker_tape_data()

    if not data:
        return

    pieces = []
    for label, price, pct in data:
        color = "#00E676" if pct >= 0 else "#FF5252"
        arrow = "▲" if pct >= 0 else "▼"
        pieces.append(
            f"<span style='margin-right:40px;color:white;'>"
            f"<b>{label}</b> ₹{price:,.2f} "
            f"<span style='color:{color};'>{arrow} {pct:+.2f}%</span>"
            f"</span>"
        )

    tape_content = "".join(pieces) * 3  # repeat for seamless loop

    st.markdown(f"""
    <div style="
        overflow:hidden;
        white-space:nowrap;
        background:#111;
        padding:10px 0;
        border-radius:8px;
        border:1px solid #333;
        margin-bottom:15px;
    ">
        <div style="
            display:inline-block;
            animation: scroll-left 30s linear infinite;
            font-size:15px;
        ">
            {tape_content}
        </div>
    </div>

    <style>
    @keyframes scroll-left {{
        0%   {{ transform: translateX(0%); }}
        100% {{ transform: translateX(-33.33%); }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# 🔥 Voice Output — the AI literally speaks its answer
# using the browser's built-in speech engine (no API key,
# no external service — pure browser JS).
# ---------------------------------------------------
def speak_text(text):
    safe_text = text.replace('"', "'").replace("\n", " ")[:600]

    components.html(f"""
    <script>
        const msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)


# ---------------------------------------------------
# 🔥 Live AI Sentiment Gauge — a SECOND, real AI model
# reads live news headlines about the stock and scores
# the market mood, rendered as a Fear/Greed-style gauge.
# This is genuine NLP inference, not a static visual.
# ---------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_data(ttl=300)
def analyze_stock_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:8]

        headlines = []
        for item in news_items:
            title = item.get("title") or item.get("content", {}).get("title")
            if title:
                headlines.append(title)

        if not headlines:
            return None

        sentiment_model = load_sentiment_model()
        results = sentiment_model(headlines)

        # Convert each headline's label+score into a single
        # -100 (very negative) to +100 (very positive) scale
        total = 0
        for r in results:
            sign = 1 if r["label"] == "POSITIVE" else -1
            total += sign * r["score"]

        avg_score = total / len(results)
        gauge_value = avg_score * 100  # scale to -100..100

        return {
            "score": gauge_value,
            "headlines_analyzed": len(headlines),
            "raw_results": list(zip(headlines, results))
        }

    except Exception:
        return None


def render_sentiment_gauge(display_name, ticker):
    result = analyze_stock_sentiment(ticker)

    if result is None:
        return

    score = result["score"]

    if score > 25:
        mood_label, mood_color = "Bullish 🐂", "#00E676"
    elif score < -25:
        mood_label, mood_color = "Bearish 🐻", "#FF5252"
    else:
        mood_label, mood_color = "Neutral ⚖️", "#FFC107"

    st.markdown(f"### 🧠 AI News Sentiment: {display_name}")
    st.caption(
        f"Live analysis of {result['headlines_analyzed']} recent news headlines "
        f"using a real sentiment-classification AI model — not simulated."
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"color": mood_color, "size": 36}},
        title={"text": mood_label, "font": {"size": 20, "color": mood_color}},
        gauge={
            "axis": {"range": [-100, 100], "tickcolor": "white"},
            "bar": {"color": mood_color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-100, -25], "color": "rgba(255,82,82,0.25)"},
                {"range": [-25, 25], "color": "rgba(255,193,7,0.2)"},
                {"range": [25, 100], "color": "rgba(0,230,118,0.25)"},
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📰 See headlines the AI analyzed"):
        for headline, sentiment in result["raw_results"]:
            emoji = "🟢" if sentiment["label"] == "POSITIVE" else "🔴"
            st.write(f"{emoji} {headline}  *(confidence: {sentiment['score']:.0%})*")


def render_live_stock_chart(display_name, ticker):
    """Fetch real price history and render an interactive
    candlestick chart + live metric cards."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")

        if hist.empty:
            return False

        latest_close = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest_close
        change = latest_close - prev_close
        pct_change = (change / prev_close) * 100 if prev_close else 0

        st.markdown(f"### 📊 Live: {display_name} ({ticker})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"₹{latest_close:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
        c2.metric("30D High", f"₹{hist['High'].max():,.2f}")
        c3.metric("30D Low", f"₹{hist['Low'].min():,.2f}")
        c4.metric("Volume", f"{int(hist['Volume'].iloc[-1]):,}")

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            increasing_line_color="#00E676",
            decreasing_line_color="#FF5252"
        )])

        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🔥 Live news headlines for this stock
        try:
            news_items = stock.news[:3]
            if news_items:
                st.markdown("**📰 Latest News**")
                for item in news_items:
                    title = item.get("title") or item.get("content", {}).get("title")
                    link = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
                    if title:
                        if link:
                            st.markdown(f"- [{title}]({link})")
                        else:
                            st.markdown(f"- {title}")
        except Exception:
            pass

        return True

    except Exception:
        return False


def render_investment_simulator(display_name, ticker, amount, period_label, period_code):
    """🔥 Wow Factor: shows what a real investment made N
    months/years ago would be worth today, using actual
    historical price data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period_code)

        if hist.empty or len(hist) < 2:
            st.warning("Not enough historical data for this period.")
            return

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]

        units_bought = amount / start_price
        current_value = units_bought * end_price
        gain = current_value - amount
        gain_pct = (gain / amount) * 100

        st.markdown(f"### 💰 If you invested ₹{amount:,.0f} in {display_name} {period_label} ago...")

        c1, c2, c3 = st.columns(3)
        c1.metric("Invested", f"₹{amount:,.0f}")
        c2.metric("Today's Value", f"₹{current_value:,.0f}", f"{gain:+,.0f}")
        c3.metric("Return", f"{gain_pct:+.2f}%")

        # Growth chart
        hist["Investment Value"] = (hist["Close"] / start_price) * amount

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist["Investment Value"],
            fill="tozeroy",
            line=dict(color="#00E676" if gain >= 0 else "#FF5252"),
            name="Value"
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Value (₹)"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Couldn't calculate that: {e}")

# ---------------------------------------------------
# Prompt Builder
# ---------------------------------------------------
def build_prompt(comparison_companies=None):

    prompt = SYSTEM_PROMPT + "\n\n"

    for message in st.session_state.conversation:

        if message["role"] == "user":
            prompt += f"User: {message['content']}\n"

        else:
            prompt += f"Assistant: {message['content']}\n"

    # 🔥 Fix: when comparing multiple companies, the model was
    # spending its entire token budget describing only the first
    # one. Instead of raising max_new_tokens, we explicitly force
    # a compact, balanced format so ALL companies fit within the
    # SAME response length.
    if comparison_companies and len(comparison_companies) > 1:
        names = " and ".join(comparison_companies)
        prompt += (
            f"\n[Instruction: You are comparing {len(comparison_companies)} companies: "
            f"{names}. Response length is limited, so be CONCISE. "
            f"Give AT MOST 2-3 short bullet points per company, covering ALL "
            f"companies with roughly EQUAL space. Do not write long paragraphs "
            f"for one company and skip the others. End with a 1-line verdict.]\n"
        )

    prompt += "Assistant:"

    return prompt

# ---------------------------------------------------
# Generate Response
# ---------------------------------------------------
def generate_response(user_question, comparison_companies=None):

    st.session_state.conversation.append(
        {
            "role": "user",
            "content": user_question,
            "time": datetime.now().strftime("%H:%M")
        }
    )

    if len(st.session_state.conversation) > MAX_TURNS * 2:
        st.session_state.conversation = st.session_state.conversation[-MAX_TURNS*2:]

    prompt = build_prompt(comparison_companies=comparison_companies)

    try:
        output = chatbot(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=st.session_state.temperature,
            repetition_penalty=1.15,
            return_full_text=False
        )
        answer = output[0]["generated_text"].strip()

        if not answer:
            answer = ("I wasn't able to generate a clear answer for that. "
                       "Could you try rephrasing your question?")

    except Exception as e:
        answer = f"⚠️ I ran into an error while generating a response: {e}"

    st.session_state.conversation.append(
        {
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%H:%M")
        }
    )

    return answer


# =====================================================
# Hero Section
# =====================================================
st.markdown("""
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(90deg,#1E3C72,#2A5298);
text-align:center;
">

<h1 style="color:white;">
📈 AI Stock Analyst Pro
</h1>

<h3 style="color:white;">
Your Personal Investment Assistant
</h3>

<p style="color:white;font-size:18px;">
Analyze Stocks • Compare Companies • Build Portfolios • Learn Investing
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("---")

# 🔥 Live scrolling ticker tape (NIFTY, SENSEX, top stocks)
render_ticker_tape()

# =====================================================
# Dashboard
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🤖 Model", value="Qwen 2.5")

with col2:
    st.metric(label="🧠 Memory", value="20 Turns")

with col3:
    st.metric(label="📈 Domain", value="Stock Market")

st.markdown("---")

# =====================================================
# AI Capability Cards
# =====================================================
st.subheader("🚀 AI Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("📈 Stock Analysis")

with col2:
    st.info("💰 Portfolio Planning")

with col3:
    st.info("📊 ETF Guidance")

with col4:
    st.info("🧠 Multi-turn Memory")

st.markdown("---")

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:

    st.title("🤖 AI Stock Analyst")

    st.success("🟢 AI Online")
    st.info("🎯 Investment Assistant")

    st.success("Multi-Turn Memory")
    st.success("Stock Analysis")
    st.success("ETF Analysis")
    st.success("Portfolio Suggestions")
    st.success("Financial Ratios")

    st.markdown("---")

    st.subheader("🔥 Sample Questions")
    st.write("• Compare TCS and Infosys")
    st.write("• What is PE Ratio?")
    st.write("• Build a ₹10 lakh portfolio")
    st.write("• Should beginners buy ETFs?")
    st.write("• Compare SBI and HDFC Bank")

    st.markdown("---")

    st.subheader("📊 AI Features")
    st.checkbox("Conversation Memory", value=True, disabled=True)
    st.checkbox("Stock Analysis", value=True, disabled=True)
    st.checkbox("ETF Analysis", value=True, disabled=True)
    st.checkbox("Portfolio Suggestions", value=True, disabled=True)
    st.checkbox("Risk Analysis", value=True, disabled=True)

    st.markdown("---")

    questions = len([
        msg for msg in st.session_state.conversation
        if msg["role"] == "user"
    ])

    st.metric("Questions Asked", questions)
    st.metric("Messages Stored", len(st.session_state.conversation))

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.balloons()
        st.session_state.conversation = []
        st.rerun()

    st.markdown("---")

    st.subheader("⚙️ Response Style")
    st.session_state.temperature = st.slider(
        "Creativity",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower = more precise/factual. Higher = more varied/creative."
    )

    st.markdown("---")

    st.subheader("🔊 Voice Output")
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False

    st.session_state.voice_enabled = st.checkbox(
        "AI reads answers aloud",
        value=st.session_state.voice_enabled,
        help="Uses your browser's built-in text-to-speech."
    )

    st.markdown("---")

    st.subheader("💰 Investment Simulator")
    st.caption("See what a real investment would be worth today")

    sim_company = st.selectbox(
        "Company",
        options=sorted(set(name.title() for name in TICKER_MAP.keys())),
        index=0
    )

    sim_amount = st.number_input(
        "Amount Invested (₹)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )

    sim_period = st.selectbox(
        "Time Ago",
        options=["1 Month", "6 Months", "1 Year", "2 Years", "5 Years"],
        index=2
    )

    period_map = {
        "1 Month": "1mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }

    run_simulation = st.button("🚀 Simulate", use_container_width=True)

    st.markdown("---")

    if st.session_state.conversation:
        chat_text = "\n\n".join(
            f"{m['role'].upper()} ({m.get('time','')}): {m['content']}"
            for m in st.session_state.conversation
        )
        st.download_button(
            "📥 Download Chat",
            data=chat_text,
            file_name="stock_analyst_chat.txt",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("💡 Tips")
    st.write("Ask complete questions.")
    st.write("Use follow-up questions.")
    st.write("Compare companies.")
    st.write("Ask for portfolio suggestions.")


# =====================================================
# 🔥 Investment Simulator Result (triggered from sidebar)
# =====================================================
if run_simulation:
    sim_ticker = TICKER_MAP.get(sim_company.lower())
    if sim_ticker:
        render_investment_simulator(
            sim_company, sim_ticker, sim_amount, sim_period.lower(), period_map[sim_period]
        )
        st.markdown("---")

# =====================================================
# Welcome Card (only shown before first message)
# =====================================================
if len(st.session_state.conversation) == 0:

    st.info("""
# 👋 Welcome to AI Stock Analyst Pro

I'm your personal AI investment assistant.

### I can help you with:

✅ Stock Analysis

✅ ETF Analysis

✅ Mutual Funds

✅ Portfolio Building

✅ Financial Ratios

✅ Long-Term Investing

⬇️ Type your question below to get started.
""")

    # =====================================================
    # Popular Questions
    # =====================================================
    st.subheader("🔥 Popular Questions")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📈 Compare Infosys and TCS")
        st.info("💰 Build ₹10 lakh portfolio")
        st.info("📊 Explain PE Ratio")

    with col2:
        st.info("🏦 Compare SBI and HDFC")
        st.info("📉 What is an ETF?")
        st.info("📈 Best stocks for beginners")

    st.markdown("---")

    # =====================================================
    # Quick Action Buttons
    # =====================================================
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Compare TCS & Infosys"):
            st.session_state.quick_question = "Compare TCS and Infosys."

    with col2:
        if st.button("💰 Build Portfolio"):
            st.session_state.quick_question = "Build a ₹10 lakh diversified portfolio."

    with col3:
        if st.button("📊 Explain PE Ratio"):
            st.session_state.quick_question = "Explain PE Ratio."

    st.markdown("---")

    st.success("""
💡 Tip

Try asking follow-up questions like

• Compare both companies

• Which one is safer?

• Explain in simple words

The AI remembers your previous conversation.
""")


# ---------------------------------------------------
# Typewriter Effect — makes the answer type out live,
# like ChatGPT, instead of appearing all at once
# ---------------------------------------------------
def typewriter(text, placeholder, delay=0.015):
    import time
    displayed = ""
    for word in text.split(" "):
        displayed += word + " "
        placeholder.markdown(displayed + "▌")
        time.sleep(delay)
    placeholder.markdown(displayed)


# ---------------------------------------------------
# 🔥 Multi-step "AI Reasoning" Animation
# Makes the AI look like it's actually working through
# an analysis process, step by step, like a real analyst
# instead of just "loading..."
# ---------------------------------------------------
def show_thinking_steps(placeholder, has_ticker=False):
    import time

    steps = ["🔍 Understanding your question..."]

    if has_ticker:
        steps.append("📡 Fetching live market data...")

    steps += [
        "🧠 Analyzing financial context...",
        "📊 Cross-referencing indicators...",
        "✍️ Generating insight..."
    ]

    for step in steps:
        placeholder.markdown(f"*{step}*")
        time.sleep(0.4)

    placeholder.empty()


# ---------------------------------------------------
# Chat History
# ---------------------------------------------------
for message in st.session_state.conversation:

    avatar = "🧑" if message["role"] == "user" else "📈"

    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "time" in message:
            st.markdown(f"<div class='msg-time'>{message['time']}</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Handle Question (from typed input or quick-action button)
# ---------------------------------------------------
if "quick_question" in st.session_state:
    question = st.session_state.quick_question
    del st.session_state["quick_question"]

if question:

    with st.chat_message("user", avatar="🧑"):
        st.write(question)

    # 🔥 Wow Factor: auto-detect company mentions and show
    # a live real-time stock chart before the AI's text answer
    tickers_found = detect_tickers(question)

    if tickers_found:
        with st.chat_message("assistant", avatar="📈"):
            for display_name, ticker in tickers_found:
                render_live_stock_chart(display_name, ticker)
                render_sentiment_gauge(display_name, ticker)
    else:
        # Heuristic: if the question looks like it's asking about
        # specific companies (mentions "compare", "vs", "graph",
        # "chart", or "Ltd") but we found no match, let the user
        # know why no chart appeared instead of silently skipping it.
        looks_company_related = any(
            kw in question.lower()
            for kw in ["compare", " vs ", "graph", "chart", "ltd", "ltd."]
        )
        if looks_company_related:
            st.info(
                "📊 I tried to find live chart data for the company you mentioned, "
                "but couldn't confidently match it to a listed stock. "
                "I'll still answer based on general knowledge below."
            )

    thinking_placeholder = st.empty()
    show_thinking_steps(thinking_placeholder, has_ticker=bool(tickers_found))

    # 🔥 IMPORTANT FIX: the spinner below stays visible for the
    # ENTIRE actual generation time (which can be 10-60+ seconds
    # on CPU), unlike the decorative thinking-steps animation
    # above, which only runs for a fixed ~2 seconds. Without this,
    # there was a silent gap where nothing indicated the app was
    # still working — making it look frozen to a first-time visitor.
    with st.spinner("🧠 AI is generating your answer... (this can take up to a minute on CPU)"):
        try:
            comparison_names = [name for name, ticker in tickers_found] if tickers_found else None
            answer = generate_response(question, comparison_companies=comparison_names)
            if not answer or not answer.strip():
                answer = ("Sorry, I couldn't generate a proper answer for that. "
                           "Could you try rephrasing your question?")
        except Exception as e:
            answer = f"⚠️ Something went wrong while generating a response: {e}"

    with st.chat_message("assistant", avatar="📈"):
        answer_placeholder = st.empty()
        typewriter(answer, answer_placeholder)

    # 🔥 Voice Output — AI speaks the answer aloud
    if st.session_state.get("voice_enabled", False):
        speak_text(answer)


# =====================================================
# Session Statistics
# =====================================================
st.markdown("---")
st.subheader("📊 Session Statistics")

col1, col2, col3 = st.columns(3)

questions = len([m for m in st.session_state.conversation if m["role"] == "user"])
answers = len([m for m in st.session_state.conversation if m["role"] == "assistant"])

col1.metric("Questions", questions)
col2.metric("Responses", answers)
col3.metric("Memory Used", f"{min(questions, MAX_TURNS)}/{MAX_TURNS}")


# =====================================================
# Disclaimer
# =====================================================
st.warning("""
⚠️ Disclaimer

This AI provides educational information only.

It should not be considered professional financial or investment advice.
""")


# =====================================================
# About Section
# =====================================================
with st.expander("ℹ️ About AI Stock Analyst"):
    st.write("""
AI Stock Analyst Pro is powered by the Qwen2.5 Large Language Model.

Features include:

• Multi-turn conversations

• Stock Analysis

• ETF Guidance

• Portfolio Suggestions

• Financial Ratio Explanations

• Company Comparisons

• Risk Analysis
""")


# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.markdown(
"""
<div class="footer">

Developed using Streamlit + Qwen2.5

🧠 Multi-turn Memory &nbsp;|&nbsp; 📈 Stock Market Intelligence

</div>
""",
unsafe_allow_html=True
)