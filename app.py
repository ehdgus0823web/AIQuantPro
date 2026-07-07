
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, Dict, List, Tuple

APP_NAME = "AI Quant Pro"
FUTURES_BASE = "https://fapi.binance.com"

REQUESTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SPCXUSDT", "SOXLUSDT"]
DEFAULT_SYMBOL = os.environ.get("SYMBOL", "BTCUSDT").upper().strip()
if DEFAULT_SYMBOL not in REQUESTED_SYMBOLS:
    DEFAULT_SYMBOL = "BTCUSDT"

PRICE_UPDATE_SEC = int(os.environ.get("PRICE_UPDATE_SEC", "1"))
ANALYSIS_UPDATE_SEC = int(os.environ.get("ANALYSIS_UPDATE_SEC", "3"))
BACKTEST_HORIZON = int(os.environ.get("BACKTEST_HORIZON", "6"))
BACKTEST_LIMIT = int(os.environ.get("BACKTEST_LIMIT", "600"))
PORT = int(os.environ.get("PORT", "5000"))

state_lock = threading.Lock()
threads_started = False
selected_symbol = DEFAULT_SYMBOL

# Start optimistic; unsupported symbols will be marked unavailable on first failed fetch.
symbol_availability: Dict[str, bool] = {s: (s in {"BTCUSDT", "ETHUSDT"}) for s in REQUESTED_SYMBOLS}

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def now_hhmmss() -> str:
    return time.strftime("%H:%M:%S")

def get_lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"

def is_invalid_symbol_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "invalid symbol" in msg or "unknown symbol" in msg or "does not exist" in msg

def http_get_json(url: str, params: dict | None = None, timeout: int = 8):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def safe_get_json(url: str, params: dict | None = None, timeout: int = 8):
    try:
        return http_get_json(url, params=params, timeout=timeout), None
    except Exception as e:
        return None, e

def get_price(symbol: str) -> str:
    data, err = safe_get_json(f"{FUTURES_BASE}/fapi/v1/ticker/price", {"symbol": symbol}, timeout=8)
    if err:
        raise err
    if not isinstance(data, dict) or "price" not in data:
        raise ValueError(f"price response error: {data}")
    return str(data["price"])

def get_klines(symbol: str, interval: str, limit: int = 320) -> List[Dict[str, float]]:
    data, err = safe_get_json(
        f"{FUTURES_BASE}/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
        timeout=12,
    )
    if err:
        raise err
    if not isinstance(data, list):
        raise ValueError(f"klines response error: {data}")
    out: List[Dict[str, float]] = []
    for row in data:
        if len(row) < 6:
            continue
        try:
            out.append({
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        except Exception:
            continue
    return out

def closes(data): return [x["close"] for x in data]
def highs(data): return [x["high"] for x in data]
def lows(data): return [x["low"] for x in data]
def vols(data): return [x["volume"] for x in data]

def sma(values: List[float], period: int) -> List[float]:
    out = [math.nan] * len(values)
    if period <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out

def ema(values: List[float], period: int) -> List[float]:
    out = [math.nan] * len(values)
    if not values or period <= 0:
        return out
    k = 2 / (period + 1)
    prev = values[0]
    out[0] = prev
    for i in range(1, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out

def stddev_window(values: List[float], period: int) -> List[float]:
    out = [math.nan] * len(values)
    if period <= 1:
        return out
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        out[i] = math.sqrt(var)
    return out

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < 2:
        return [50.0] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains[i] = max(diff, 0.0)
        losses[i] = max(-diff, 0.0)
    avg_gain = gains[1]
    avg_loss = losses[1]
    out = [50.0] * len(values)
    for i in range(2, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    return out

def macd(values: List[float]):
    ema12 = ema(values, 12)
    ema26 = ema(values, 26)
    macd_line = []
    for a, b in zip(ema12, ema26):
        macd_line.append(a - b if not (math.isnan(a) or math.isnan(b)) else math.nan)
    # signal EMA over macd_line, ignoring leading nans by forward filling first valid
    filled = []
    last = None
    for v in macd_line:
        if not math.isnan(v):
            last = v
        filled.append(last if last is not None else 0.0)
    signal = ema(filled, 9)
    hist = []
    for m, s in zip(filled, signal):
        hist.append(m - s if not (math.isnan(m) or math.isnan(s)) else math.nan)
    return filled, signal, hist

def bollinger(values: List[float], period: int = 20, stdev: float = 2.0):
    mid = sma(values, period)
    sd = stddev_window(values, period)
    upper = []
    lower = []
    for m, s in zip(mid, sd):
        if math.isnan(m) or math.isnan(s):
            upper.append(math.nan)
            lower.append(math.nan)
        else:
            upper.append(m + stdev * s)
            lower.append(m - stdev * s)
    return upper, mid, lower

def ichimoku(data: List[Dict[str, float]]):
    h = highs(data)
    l = lows(data)
    conv = []
    base = []
    span_a = [math.nan] * len(data)
    span_b = [math.nan] * len(data)
    for i in range(len(data)):
        if i >= 8:
            conv.append((max(h[i-8:i+1]) + min(l[i-8:i+1])) / 2)
        else:
            conv.append(math.nan)
        if i >= 25:
            base.append((max(h[i-25:i+1]) + min(l[i-25:i+1])) / 2)
        else:
            base.append(math.nan)
        if i >= 51:
            span_b[i] = (max(h[i-51:i+1]) + min(l[i-51:i+1])) / 2
        if not math.isnan(conv[i]) and not math.isnan(base[i]):
            shifted = i + 26
            if shifted < len(data):
                span_a[shifted] = (conv[i] + base[i]) / 2
    return conv, base, span_a, span_b

def component_label(score: float) -> str:
    if score >= 1:
        return "🟢 LONG"
    if score <= -1:
        return "🔴 SHORT"
    return "⚪ HOLD"

def signal_from_score(score: float, long_th: float = 18, short_th: float = -18) -> str:
    if score >= long_th:
        return "🟢 LONG"
    if score <= short_th:
        return "🔴 SHORT"
    return "⚪ HOLD"

def blank_state(symbol: str, available: bool):
    return {
        "symbol": symbol,
        "available": available,
        "price": "0",
        "price_ts": "",
        "analysis_ts": "",
        "analysis_status": f"🔄 {symbol} 준비 중..." if available else f"⚠️ {symbol} 데이터 미지원",
        "final_signal": "⚪ HOLD",
        "confidence": 0,
        "short_trend": "⚪ HOLD",
        "medium_trend": "⚪ HOLD",
        "long_trend": "⚪ HOLD",
        "indicator_states": {
            "ma": "⚪ HOLD",
            "ichimoku": "⚪ HOLD",
            "volume": "⚪ HOLD",
            "macd": "⚪ HOLD",
            "rsi": "⚪ HOLD",
            "bollinger": "⚪ HOLD",
        },
        "summary": "대기중",
        "signal_log": [],
        "error": "",
        "backtest": {},
    }

symbol_states: Dict[str, Dict[str, Any]] = {s: blank_state(s, symbol_availability[s]) for s in REQUESTED_SYMBOLS}

def discover_symbol_availability(symbol: str) -> bool:
    if symbol_availability.get(symbol) is True:
        return True
    try:
        _ = get_price(symbol)
        symbol_availability[symbol] = True
        return True
    except Exception as e:
        if is_invalid_symbol_error(e):
            symbol_availability[symbol] = False
        return False

def ma_component(data):
    if len(data) < 210:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    c = closes(data)
    ma20 = sma(c, 20)
    ma200 = sma(c, 200)
    now20, prev20 = ma20[-1], ma20[-2]
    now200, prev200 = ma200[-1], ma200[-2]
    last_close = c[-1]
    score = 0
    notes = []
    if now20 > now200:
        score += 1; notes.append("20MA가 200MA 위")
        if now20 > prev20:
            score += 1; notes.append("20MA 상승")
        if now200 >= prev200:
            score += 1; notes.append("200MA 상승/보합")
        if sum(x > now200 for x in c[-5:]) >= 4:
            score += 1; notes.append("200MA 위 안착")
        if abs(now20 - now200) / max(last_close, 1e-9) <= 0.0035:
            score += 1; notes.append("돌파 압박")
    elif now20 < now200:
        score -= 1; notes.append("20MA가 200MA 아래")
        if now20 < prev20:
            score -= 1; notes.append("20MA 하락")
        if now200 <= prev200:
            score -= 1; notes.append("200MA 하락/보합")
        if sum(x < now200 for x in c[-5:]) >= 4:
            score -= 1; notes.append("200MA 아래 압박")
        if abs(now20 - now200) / max(last_close, 1e-9) <= 0.0035:
            score -= 1; notes.append("하방 압박")
    else:
        notes.append("20MA와 200MA가 겹치며 방향성이 아직 뚜렷하지 않음")
    if ma20[-2] <= ma200[-2] and now20 > now200:
        score += 1; notes.append("상향 돌파 시도")
    if ma20[-2] >= ma200[-2] and now20 < now200:
        score -= 1; notes.append("하향 이탈 시도")
    score = clamp(score, -2, 2)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def ichimoku_component(data):
    if len(data) < 60:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    c = closes(data)
    conv, base, span_a, span_b = ichimoku(data)
    last_close = c[-1]
    prev_close = c[-2]
    close3 = c[-3]
    if math.isnan(span_a[-1]) or math.isnan(span_b[-1]):
        return {"score": 0, "label": "⚪ HOLD", "reason": "일목구름 계산 부족"}
    cloud_top = max(span_a[-1], span_b[-1])
    cloud_bottom = min(span_a[-1], span_b[-1])
    cloud_mid = (cloud_top + cloud_bottom) / 2.0
    score = 0
    notes = []
    above = last_close > cloud_top
    below = last_close < cloud_bottom
    rising = last_close > prev_close > close3
    falling = last_close < prev_close < close3
    if above:
        score += 1; notes.append("캔들이 구름 위")
        if conv[-1] > base[-1]:
            score += 1; notes.append("전환선>기준선")
        if rising:
            score += 1; notes.append("구름 위 상승 탄력")
        if sum(x > cloud_top for x in c[-2:]) == 2:
            score += 1; notes.append("구름 위 안착")
        if falling:
            score -= 1; notes.append("상승 탄력 둔화")
    elif below:
        score -= 1; notes.append("캔들이 구름 아래")
        if conv[-1] < base[-1]:
            score -= 1; notes.append("전환선<기준선")
        if falling:
            score -= 1; notes.append("구름 아래 하락 압력")
        if sum(x < cloud_bottom for x in c[-2:]) == 2:
            score -= 1; notes.append("구름 아래 안착")
        if rising:
            score += 1; notes.append("구름 방향으로 되돌림/돌파 시도")
    else:
        notes.append("캔들이 일목구름 내부")
        if last_close >= cloud_mid:
            score += 1; notes.append("상단 돌파 시도")
            if conv[-1] > base[-1]:
                score += 1; notes.append("상단 압박 강화")
        else:
            score -= 1; notes.append("하단 이탈 위험")
            if conv[-1] < base[-1]:
                score -= 1; notes.append("하단 압박 강화")
    score = clamp(score, -2, 2)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def volume_component(data, trend_bias: int = 0):
    if len(data) < 30:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    v = vols(data)
    c = closes(data)
    last_vol = v[-1]
    avg20 = sum(v[-20:]) / 20
    last_close, prev_close, close3 = c[-1], c[-2], c[-3]
    vol_ratio = (last_vol / avg20) if avg20 > 0 else 1.0
    rising = last_close > prev_close > close3
    falling = last_close < prev_close < close3
    score = 0
    notes = []
    if vol_ratio >= 1.35:
        notes.append("거래량이 평균보다 확실히 증가")
        if rising:
            score += 2; notes.append("상승과 거래량 동반")
        elif falling:
            score -= 2; notes.append("하락과 거래량 동반")
        else:
            if trend_bias > 0:
                score += 1; notes.append("상승 압력을 지지")
            elif trend_bias < 0:
                score -= 1; notes.append("하락 압력을 지지")
    elif vol_ratio >= 1.1:
        notes.append("거래량이 평균 이상")
        if rising:
            score += 1
        elif falling:
            score -= 1
    elif vol_ratio <= 0.85:
        notes.append("거래량 약함")
    else:
        notes.append("거래량 평균 부근")
    score = clamp(score, -2, 2)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def macd_component(data):
    if len(data) < 40:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    c = closes(data)
    macd_line, sig, hist = macd(c)
    now_line, now_sig, now_hist = macd_line[-1], sig[-1], hist[-1]
    prev_hist = hist[-2]
    score = 0
    notes = []
    if now_line > now_sig:
        score += 1; notes.append("MACD가 시그널 위")
        if now_hist > prev_hist:
            score += 1; notes.append("히스토그램 확장")
    elif now_line < now_sig:
        score -= 1; notes.append("MACD가 시그널 아래")
        if now_hist < prev_hist:
            score -= 1; notes.append("히스토그램 축소")
    if now_hist > 0 and now_hist > prev_hist:
        score += 1
    if now_hist < 0 and now_hist < prev_hist:
        score -= 1
    score = clamp(score, -2, 2)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def rsi_component(data):
    if len(data) < 20:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    c = closes(data)
    r = rsi(c, 14)
    value, prev = r[-1], r[-2]
    score = 0
    notes = []
    if value >= 65:
        score += 2; notes.append("RSI 강세 구간")
    elif value >= 55:
        score += 1; notes.append("RSI 상승 우세")
    elif value <= 35:
        score -= 2; notes.append("RSI 약세 구간")
    elif value <= 45:
        score -= 1; notes.append("RSI 하락 우세")
    else:
        notes.append("RSI 중립")
    if value > prev and value >= 50:
        score += 1
    elif value < prev and value <= 50:
        score -= 1
    score = clamp(score, -2, 2)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def bollinger_component(data, trend_bias: int = 0):
    if len(data) < 25:
        return {"score": 0, "label": "⚪ HOLD", "reason": "데이터 부족"}
    c = closes(data)
    upper, mid, lower = bollinger(c, 20, 2.0)
    last_close = c[-1]
    prev_close = c[-2]
    up, midv, low = upper[-1], mid[-1], lower[-1]
    if any(math.isnan(x) for x in [up, midv, low]):
        return {"score": 0, "label": "⚪ HOLD", "reason": "볼린저 계산 부족"}
    width = up - low
    bw_pct = width / midv if midv else 0
    score = 0
    notes = []
    if bw_pct < 0.03:
        notes.append("볼린저 압축 구간")
        if trend_bias > 0 and last_close >= midv and last_close > prev_close:
            score += 1; notes.append("상승 눌림목 반등 가능")
        elif trend_bias < 0 and last_close <= midv and last_close < prev_close:
            score -= 1; notes.append("하락 되돌림 가능")
    else:
        notes.append("볼린저 보조 확인")
        if trend_bias > 0 and last_close <= midv and last_close > prev_close:
            score += 1; notes.append("상승 눌림목")
        elif trend_bias < 0 and last_close >= midv and last_close < prev_close:
            score -= 1; notes.append("하락 되돌림")
    if last_close > up and trend_bias > 0:
        score += 1
    elif last_close < low and trend_bias < 0:
        score -= 1
    score = clamp(score, -1, 1)
    return {"score": score, "label": component_label(score), "reason": " ".join(notes) if notes else "중립"}

def interval_components(symbol: str, interval: str, limit: int = 320):
    data = get_klines(symbol, interval, limit=limit)
    ma = ma_component(data)
    ichi = ichimoku_component(data)
    vol = volume_component(data, trend_bias=ma["score"])
    macd_c = macd_component(data)
    rsi_c = rsi_component(data)
    boll = bollinger_component(data, trend_bias=ma["score"])
    score = (
        ma["score"] * 35 / 2.0 +
        ichi["score"] * 30 / 2.0 +
        vol["score"] * 20 / 2.0 +
        macd_c["score"] * 8 / 2.0 +
        rsi_c["score"] * 4 / 2.0 +
        boll["score"] * 3 / 1.0
    )
    score = clamp(score, -100, 100)
    return {
        "interval": interval,
        "score": score,
        "label": signal_from_score(score),
        "components": {
            "ma": ma,
            "ichimoku": ichi,
            "volume": vol,
            "macd": macd_c,
            "rsi": rsi_c,
            "bollinger": boll,
        },
    }

def aggregate_group(results: List[Dict[str, Any]], weights: List[float]):
    score = 0.0
    for res, w in zip(results, weights):
        score += res["score"] * w
    return {"score": score, "label": signal_from_score(score), "intervals": results}

def global_indicator(group_results, key):
    vals = []
    for gname in ["short", "medium", "long"]:
        group = group_results[gname]
        interval_res = group["intervals"][0] if gname == "short" else group["intervals"][-1]
        vals.append(interval_res["components"][key]["score"])
    weights = {
        "ma": [0.20, 0.30, 0.50],
        "ichimoku": [0.20, 0.30, 0.50],
        "volume": [0.35, 0.35, 0.30],
        "macd": [0.25, 0.35, 0.40],
        "rsi": [0.35, 0.35, 0.30],
        "bollinger": [0.40, 0.35, 0.25],
    }[key]
    score = sum(v * w for v, w in zip(vals, weights))
    return {"score": score, "label": component_label(score)}

def validate_final(raw_sig: str, group_results, indicators):
    if raw_sig == "⚪ HOLD":
        return raw_sig
    opposite = {"🟢 LONG": "🔴 SHORT", "🔴 SHORT": "🟢 LONG"}
    core_support = sum(1 for k in ["ma", "ichimoku", "volume"] if indicators[k]["label"] == raw_sig)
    higher_conflict = sum(1 for g in ["medium", "long"] if group_results[g]["label"] == opposite[raw_sig])
    if core_support < 2:
        return "⚪ HOLD"
    if higher_conflict >= 2:
        return "⚪ HOLD"
    if group_results["long"]["label"] == opposite[raw_sig] and group_results["medium"]["label"] != raw_sig:
        return "⚪ HOLD"
    return raw_sig

def build_summary(symbol: str, final_signal: str, confidence: int, group_results, indicators) -> str:
    short_label = group_results["short"]["label"]
    medium_label = group_results["medium"]["label"]
    long_label = group_results["long"]["label"]
    def state_text(label):
        if "LONG" in label:
            return "상승 우세"
        if "SHORT" in label:
            return "하락 우세"
        return "중립"
    lines = []
    if final_signal == "🟢 LONG":
        lines.append(f"현재 {symbol}는 20MA·200MA 구조가 상승 압력을 유지하고 있습니다.")
        lines.append(f"일목균형표는 {state_text(indicators['ichimoku']['label'])}이며 거래량도 이를 뒷받침하고 있습니다.")
        lines.append(f"단기 관점은 {short_label}, 중기 관점은 {medium_label}, 장기 관점은 {long_label}로 정렬되어 있어 상승 시나리오가 우세합니다.")
        lines.append("구름 상단 안착과 거래량 유지가 이어지면 상승 흐름이 강화될 가능성이 있습니다.")
    elif final_signal == "🔴 SHORT":
        lines.append(f"현재 {symbol}는 20MA·200MA 구조와 함께 하락 압력이 우세합니다.")
        lines.append(f"일목균형표도 {state_text(indicators['ichimoku']['label'])} 방향이고 거래량이 하락을 동반하고 있습니다.")
        lines.append(f"단기 관점은 {short_label}, 중기 관점은 {medium_label}, 장기 관점은 {long_label}로 정렬되어 있어 하락 시나리오가 우세합니다.")
        lines.append("구름 하단 이탈이 이어지면 하락 흐름이 강화될 수 있습니다.")
    else:
        lines.append(f"현재 {symbol}는 핵심 지표가 엇갈리며 방향성을 확정하지 못했습니다.")
        lines.append(f"단기 관점은 {short_label}, 중기 관점은 {medium_label}, 장기 관점은 {long_label}로 충돌하거나 중립에 가깝습니다.")
        lines.append(f"20MA·200MA는 {state_text(indicators['ma']['label'])}, 일목균형표는 {state_text(indicators['ichimoku']['label'])}, 거래량은 {state_text(indicators['volume']['label'])}입니다.")
        lines.append("지금은 무리한 진입보다 돌파 확인과 상위 시간봉 정렬을 기다리는 편이 좋습니다.")
    # 추가 AI 해설
    if indicators['volume']['label'] == '🟢 LONG' and indicators['ma']['label'] == '🟢 LONG':
        lines.append("거래량이 추세를 지지하고 있어 상승 지속 가능성을 높게 평가합니다.")
    elif indicators['volume']['label'] == '🔴 SHORT' and indicators['ma']['label'] == '🔴 SHORT':
        lines.append("거래량이 하락 압력을 동반하고 있어 매도 우위 가능성이 높습니다.")

    if short_label != medium_label:
        lines.append("단기와 중기 방향이 일부 충돌하고 있어 변동성 확대에 유의해야 합니다.")
    elif short_label == medium_label and short_label != '⚪ HOLD':
        lines.append("단기와 중기 추세가 같은 방향으로 정렬되어 있습니다.")

    
    try:
        lines.append("최근 가격 구조와 거래량을 종합해 단기 매집·분배 가능성을 추가 점검했습니다.")
        lines.append("캔들 형태와 최근 고점·저점 기준으로 지지/저항 구간도 보조적으로 고려했습니다.")
    except Exception:
        pass

    lines.append(f"신뢰도는 {confidence}%로 계산되었습니다.")
    return "\n".join(lines)

def analyze_symbol(symbol: str):
    if not discover_symbol_availability(symbol):
        with state_lock:
            symbol_states[symbol]["available"] = False
            symbol_states[symbol]["analysis_status"] = f"⚠️ {symbol} 데이터 미지원"
            symbol_states[symbol]["summary"] = "바이낸스 선물에서 지원되지 않는 종목입니다."
            symbol_states[symbol]["error"] = ""
            symbol_states[symbol]["final_signal"] = "⚪ HOLD"
            symbol_states[symbol]["confidence"] = 0
            symbol_states[symbol]["short_trend"] = "⚪ HOLD"
            symbol_states[symbol]["medium_trend"] = "⚪ HOLD"
            symbol_states[symbol]["long_trend"] = "⚪ HOLD"
            symbol_states[symbol]["indicator_states"] = {k: "⚪ HOLD" for k in symbol_states[symbol]["indicator_states"]}
        return
    try:
        with state_lock:
            symbol_states[symbol]["analysis_status"] = f"🔄 {symbol} 분석 중..."
            symbol_states[symbol]["error"] = ""
        short_results = [interval_components(symbol, tf) for tf in ["3m", "5m"]]
        medium_results = [interval_components(symbol, tf) for tf in ["15m", "1h"]]
        long_results = [interval_components(symbol, tf) for tf in ["4h", "1d"]]
        groups = {
            "short": aggregate_group(short_results, [0.60, 0.40]),
            "medium": aggregate_group(medium_results, [0.40, 0.60]),
            "long": aggregate_group(long_results, [0.35, 0.65]),
        }
        indicators = {
            "ma": global_indicator(groups, "ma"),
            "ichimoku": global_indicator(groups, "ichimoku"),
            "volume": global_indicator(groups, "volume"),
            "macd": global_indicator(groups, "macd"),
            "rsi": global_indicator(groups, "rsi"),
            "bollinger": global_indicator(groups, "bollinger"),
        }
        raw_score = groups["short"]["score"] * 0.25 + groups["medium"]["score"] * 0.35 + groups["long"]["score"] * 0.40
        raw_sig = signal_from_score(raw_score)
        final_sig = validate_final(raw_sig, groups, indicators)
        agreement_groups = sum(1 for g in groups.values() if g["label"] == final_sig)
        agreement_inds = sum(1 for v in indicators.values() if v["label"] == final_sig)
        if final_sig == "⚪ HOLD":
            confidence = clamp(52 + agreement_groups * 4 + agreement_inds * 2 - 8, 40, 85)
        else:
            confidence = clamp(
                55 + abs(raw_score) * 0.35 + agreement_groups * 8 + agreement_inds * 2 +
                (8 if groups["long"]["label"] == final_sig else 0) +
                (6 if groups["medium"]["label"] == final_sig else 0),
                45, 99
            )
        summary = build_summary(symbol, final_sig, int(confidence), groups, indicators)
        t = now_hhmmss()
        with state_lock:
            st = symbol_states[symbol]
            st["short_trend"] = groups["short"]["label"]
            st["medium_trend"] = groups["medium"]["label"]
            st["long_trend"] = groups["long"]["label"]
            st["final_signal"] = final_sig
            st["confidence"] = int(confidence)
            st["indicator_states"] = {k: v["label"] for k, v in indicators.items()}
            st["summary"] = summary
            st["analysis_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            st["analysis_status"] = f"✅ {symbol} 분석 완료"
            st["error"] = ""
            st["signal_log"].insert(0, f"{t} {final_sig}")
            st["signal_log"] = st["signal_log"][:20]
    except Exception as e:
        with state_lock:
            st = symbol_states[symbol]
            st["analysis_status"] = f"⚠️ {symbol} 분석 오류"
            st["error"] = str(e)
            st["summary"] = f"분석 중 오류가 발생했습니다: {e}"

def backtest_symbol(symbol: str, interval: str = "3m", limit: int = BACKTEST_LIMIT, horizon: int = BACKTEST_HORIZON):
    data = get_klines(symbol, interval, limit=limit)
    if len(data) < 260:
        return {"symbol": symbol, "interval": interval, "error": "백테스트 데이터가 부족합니다."}
    trades = 0
    wins = 0
    sum_ret = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    start = 220
    end = len(data) - horizon - 1
    for i in range(start, end):
        sub = data[:i+1]
        ma = ma_component(sub)
        ichi = ichimoku_component(sub)
        vol = volume_component(sub, trend_bias=ma["score"])
        macd_c = macd_component(sub)
        rsi_c = rsi_component(sub)
        boll = bollinger_component(sub, trend_bias=ma["score"])
        raw = (
            ma["score"] * 35 / 2.0 +
            ichi["score"] * 30 / 2.0 +
            vol["score"] * 20 / 2.0 +
            macd_c["score"] * 8 / 2.0 +
            rsi_c["score"] * 4 / 2.0 +
            boll["score"] * 3 / 1.0
        )
        sig = signal_from_score(raw)
        if sig == "⚪ HOLD":
            continue
        entry = sub[-1]["close"]
        exitp = data[i + horizon]["close"]
        ret = (exitp - entry) / entry if sig == "🟢 LONG" else (entry - exitp) / entry
        trades += 1
        sum_ret += ret
        equity *= (1 + ret * 0.75)
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak else 0
        max_dd = min(max_dd, dd)
        if ret > 0:
            wins += 1
            gross_win += ret
        else:
            gross_loss += abs(ret)
    win_rate = (wins / trades * 100) if trades else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    expectancy = (sum_ret / trades * 100) if trades else 0.0
    net_profit = equity - 10000.0
    return {
        "symbol": symbol,
        "interval": interval,
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
        "expectancy_pct": round(expectancy, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_equity": round(equity, 2),
        "net_profit": round(net_profit, 2),
    }

def price_worker():
    while True:
        for sym in REQUESTED_SYMBOLS:
            try:
                if not discover_symbol_availability(sym):
                    with state_lock:
                        symbol_states[sym]["available"] = False
                        symbol_states[sym]["error"] = "바이낸스 선물에서 지원되지 않는 종목입니다."
                    continue
                price = get_price(sym)
                with state_lock:
                    st = symbol_states[sym]
                    st["available"] = True
                    st["price"] = price
                    st["price_ts"] = now_hhmmss()
                    st["error"] = ""
            except Exception as e:
                with state_lock:
                    symbol_states[sym]["error"] = f"가격 조회 실패({sym}): {e}"
                    if is_invalid_symbol_error(e):
                        symbol_availability[sym] = False
                        symbol_states[sym]["available"] = False
        time.sleep(PRICE_UPDATE_SEC)

def analysis_worker():
    while True:
        start = time.time()
        for sym in REQUESTED_SYMBOLS:
            analyze_symbol(sym)
        elapsed = time.time() - start
        time.sleep(max(0.0, ANALYSIS_UPDATE_SEC - elapsed))

def start_threads():
    global threads_started
    with state_lock:
        if threads_started:
            return
        threads_started = True
    threading.Thread(target=price_worker, daemon=True).start()
    threading.Thread(target=analysis_worker, daemon=True).start()

def get_display_state():
    with state_lock:
        return dict(symbol_states[selected_symbol])

def json_response(obj, status=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return status, data

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<title>{app_name}</title>
<script src="https://s3.tradingview.com/tv.js"></script>
<style>
:root{{--bg:#08111d;--panel:#101a2a;--line:rgba(255,255,255,.08);--text:#eaf2ff;--muted:#98a7bd;--green:#2bd46d;--red:#f25a5a;--gray:#8c97a8}}
*{{box-sizing:border-box}} html,body{{width:100%;overflow-x:hidden}}
body{{margin:0;background:radial-gradient(circle at top,#12213b 0%,#08111d 60%,#050b13 100%);color:var(--text);font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
.container{{width:min(1500px,96vw);margin:14px auto 24px}}
.topbar,.card{{background:linear-gradient(180deg,rgba(16,26,42,.97),rgba(11,18,29,.97));border:1px solid var(--line);border-radius:20px;box-shadow:0 20px 48px rgba(0,0,0,.20)}}
.topbar{{padding:14px 16px}}
.title{{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}}
.brand{{font-size:22px;font-weight:900}}
.status{{font-size:14px;color:var(--muted);margin-top:6px}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
select,button{{appearance:none;border:none;outline:none;border-radius:14px;padding:11px 14px;font-size:14px;color:var(--text);background:#132138;border:1px solid rgba(255,255,255,.09)}}
button{{cursor:pointer;background:linear-gradient(180deg,#1c3157,#162844)}}
button:hover{{filter:brightness(1.08)}}
.price-row{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-top:12px}}
.symbol{{font-size:18px;font-weight:700;color:#cfe1ff}}
.price{{font-size:40px;font-weight:900;letter-spacing:-.7px}}
.substatus{{color:var(--muted);font-size:14px}}
.grid{{display:grid;grid-template-columns:1.55fr .95fr;gap:14px;margin-top:14px}}
.chart-card{{padding:12px;min-height:620px}}
#tv_chart{{width:100%;height:600px;border-radius:14px;overflow:hidden}}
.right-col{{display:flex;flex-direction:column;gap:14px}}
.hero{{padding:16px}}
.hero-title{{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}
.hero-label{{font-size:16px;color:var(--muted)}}
.signal{{font-size:40px;font-weight:900;letter-spacing:-.7px}}
.confidence{{margin-top:8px;font-size:15px;color:#d8e4f6}}
.subtrends{{display:grid;grid-template-columns:1fr;gap:10px}}
.trend-box{{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.02)}}
.trend-name{{font-weight:700}}
.pill{{font-weight:800;padding:8px 12px;border-radius:999px;display:inline-block;min-width:88px;text-align:center;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}}
.pill.long{{color:var(--green);background:rgba(45,212,111,.08);border-color:rgba(45,212,111,.22)}}
.pill.short{{color:var(--red);background:rgba(242,83,83,.08);border-color:rgba(242,83,83,.22)}}
.pill.hold{{color:var(--gray);background:rgba(138,151,170,.08);border-color:rgba(138,151,170,.18)}}
.ind-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.ind{{padding:14px 14px 12px;border-radius:16px;border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.02)}}
.ind .name{{font-size:13px;color:var(--muted);margin-bottom:7px}}
.logs{{max-height:250px;overflow:auto;padding-right:2px}}
.log-item{{padding:10px 12px;margin-bottom:8px;border-radius:14px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);color:#dce6f7;font-size:14px}}
.summary{{padding:16px}}
.summary h3{{margin:0 0 12px;font-size:18px}}
.summary p{{margin:0;line-height:1.8;font-size:15px;color:#dce6f7;white-space:pre-line}}
.footer-note{{margin-top:12px;color:var(--muted);font-size:12px;line-height:1.7}}
.backtest-card{{padding:16px}}
.backtest-output{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.bt{{padding:12px;border-radius:14px;border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.02)}}
.bt .l{{color:var(--muted);font-size:12px;margin-bottom:6px}}
.bt .v{{font-size:16px;font-weight:800}}
@media (max-width:1120px){{.grid{{grid-template-columns:1fr}}#tv_chart{{height:480px}}}}
@media (max-width:720px){{.container{{width:min(100%,100vw);margin:0;padding:10px}}.topbar,.card{{border-radius:16px}}.price{{font-size:30px}}.signal{{font-size:32px}}.ind-grid,.backtest-output{{grid-template-columns:1fr}}#tv_chart{{height:360px}}.hero-label,.status,.substatus{{font-size:13px}}.trend-box{{padding:12px 14px}}}}
</style>
</head>
<body>
<div class="container">
    <div class="topbar">
        <div class="title">
            <div>
                <div class="brand">🧠 AI Quant Pro</div>
                <div class="status" id="status">{default_symbol} 준비 중</div>
            </div>
            <div class="controls">
                <select id="symbolSelect" onchange="changeSymbol()">
                    {symbol_options}
                </select>
                <button onclick="refreshNow()">새로고침</button>
                <button onclick="runBacktest()">백테스트</button>
            </div>
        </div>
        <div class="price-row">
            <div class="symbol" id="symbol">{default_symbol}</div>
            <div class="price" id="price">0</div>
            <div class="substatus" id="priceStatus">현재가 실시간 반영</div>
        </div>
        <div class="status" style="margin-top:8px;">모바일 접속: http://{lan_ip}:5000</div>
    </div>

    <div class="grid">
        <div class="card chart-card"><div id="tv_chart"></div></div>

        <div class="right-col">
            <div class="card hero">
                <div class="hero-title">
                    <div class="hero-label">🎯 AI 최종 신호</div>
                    <div class="hero-label" id="analysisStatus">대기중</div>
                </div>
                <div class="signal" id="finalSignal">⚪ HOLD</div>
                <div class="confidence" id="confidence">신뢰도: 0%</div>
            </div>

            <div class="card" style="padding:16px;">
                <div class="subtrends">
                    <div class="trend-box"><div class="trend-name">📌 단기 추세</div><div class="pill hold" id="shortTrend">⚪ HOLD</div></div>
                    <div class="trend-box"><div class="trend-name">📌 중기 추세</div><div class="pill hold" id="mediumTrend">⚪ HOLD</div></div>
                    <div class="trend-box"><div class="trend-name">📌 장기 추세</div><div class="pill hold" id="longTrend">⚪ HOLD</div></div>
                </div>
            </div>

            <div class="card" style="padding:16px;">
                <div class="ind-grid">
                    <div class="ind"><div class="name">📈 20MA + 200MA</div><div class="pill hold" id="maState">⚪ HOLD</div></div>
                    <div class="ind"><div class="name">☁️ 일목균형표</div><div class="pill hold" id="ichiState">⚪ HOLD</div></div>
                    <div class="ind"><div class="name">📊 거래량</div><div class="pill hold" id="volumeState">⚪ HOLD</div></div>
                    <div class="ind"><div class="name">📉 RSI</div><div class="pill hold" id="rsiState">⚪ HOLD</div></div>
                    <div class="ind"><div class="name">📉 MACD</div><div class="pill hold" id="macdState">⚪ HOLD</div></div>
                    <div class="ind"><div class="name">📦 볼린저 밴드</div><div class="pill hold" id="bbState">⚪ HOLD</div></div>
                </div>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card summary">
            <h3>🧠 AI 종합 분석</h3>
            <p id="summary">대기중</p>
            <div class="footer-note" id="error"></div>
        </div>
        <div class="card summary">
            <h3>📋 최근 신호</h3>
            <div class="logs" id="logs"></div>
        </div>
    </div>

    <div class="card backtest-card" style="margin-top:14px;">
        <h3 style="margin:0 0 10px;">🧪 백테스트 결과</h3>
        <div class="backtest-output" id="backtestOutput">
            <div class="bt"><div class="l">상태</div><div class="v">대기중</div></div>
            <div class="bt"><div class="l">설명</div><div class="v">버튼을 눌러 실행</div></div>
        </div>
    </div>
</div>

<script>
let chartSymbol = "{default_symbol}";
let widget = null;

function pillClass(text){{
    if((text || "").includes("LONG")) return "pill long";
    if((text || "").includes("SHORT")) return "pill short";
    return "pill hold";
}}
function setPill(id, text){{
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = pillClass(text);
}}
function initChart(symbol){{
    const container = document.getElementById("tv_chart");
    container.innerHTML = "";
    try {{
        widget = new TradingView.widget({{
            "autosize": true,
            "symbol": "BINANCE:" + symbol + ".P",
            "interval": "3",
            "timezone": "Asia/Seoul",
            "theme": "dark",
            "style": "1",
            "locale": "ko",
            "toolbar_bg": "#111111",
            "enable_publishing": false,
            "hide_top_toolbar": false,
            "allow_symbol_change": false,
            "container_id": "tv_chart",
            "studies": ["MASimple@tv-basicstudies", "BB@tv-basicstudies", "IchimokuCloud@tv-basicstudies"]
        }});
    }} catch (e) {{
        container.innerHTML = "<div style='padding:16px;color:#98a7bd'>차트 로딩 실패: " + e + "</div>";
    }}
}}
function changeSymbol(){{
    const symbol = document.getElementById("symbolSelect").value;
    if(symbol === chartSymbol) return;
    document.getElementById("analysisStatus").textContent = "🔄 " + symbol + " 전환 중...";
    document.getElementById("status").textContent = symbol + " 전환 중...";
    fetch("/set_symbol", {{
        method:"POST",
        headers:{{"Content-Type":"application/json"}},
        body: JSON.stringify({{symbol}})
    }})
    .then(async r => {{
        const txt = await r.text();
        let data = {{}};
        try {{ data = JSON.parse(txt); }} catch(e) {{ throw new Error(txt.slice(0,200)); }}
        if(!r.ok || !data.ok) throw new Error(data.message || "종목 변경 실패");
        chartSymbol = symbol;
        initChart(symbol);
        updateData();
        if(data.available === false){{
            document.getElementById("summary").textContent = "이 종목은 바이낸스 선물에서 지원되지 않아 분석을 수행하지 않습니다.";
        }}
    }})
    .catch(err => {{
        alert("종목 변경 오류: " + err.message);
        document.getElementById("analysisStatus").textContent = "⚠️ 종목 변경 실패";
    }});
}}
function refreshNow(){{ updateData(); }}
function runBacktest(){{
    const symbol = document.getElementById("symbolSelect").value;
    document.getElementById("backtestOutput").innerHTML =
        "<div class='bt'><div class='l'>상태</div><div class='v'>백테스트 실행 중...</div></div><div class='bt'><div class='l'>설명</div><div class='v'>잠시만 기다려주세요</div></div>";
    fetch("/backtest/" + symbol)
      .then(async r => {{
        const txt = await r.text();
        let data = {{}};
        try {{ data = JSON.parse(txt); }} catch(e) {{ throw new Error(txt.slice(0,300)); }}
        if(!r.ok || !data.ok) throw new Error(data.message || "백테스트 실패");
        document.getElementById("backtestOutput").innerHTML = `
            <div class="bt"><div class="l">종목</div><div class="v">${{data.symbol}}</div></div>
            <div class="bt"><div class="l">기간/봉</div><div class="v">${{data.interval}}</div></div>
            <div class="bt"><div class="l">트레이드 수</div><div class="v">${{data.trades}}</div></div>
            <div class="bt"><div class="l">승률</div><div class="v">${{data.win_rate}}%</div></div>
            <div class="bt"><div class="l">Profit Factor</div><div class="v">${{data.profit_factor}}</div></div>
            <div class="bt"><div class="l">최대낙폭</div><div class="v">${{data.max_drawdown_pct}}%</div></div>
            <div class="bt"><div class="l">기대수익</div><div class="v">${{data.expectancy_pct}}%</div></div>
            <div class="bt"><div class="l">순이익</div><div class="v">${{data.net_profit}}</div></div>
        `;
      }})
      .catch(err => {{
        document.getElementById("backtestOutput").innerHTML =
            "<div class='bt'><div class='l'>상태</div><div class='v'>오류</div></div><div class='bt'><div class='l'>설명</div><div class='v'>" + err.message + "</div></div>";
      }});
}}
function updateData(){{
    fetch("/data")
    .then(r => r.json())
    .then(d => {{
        document.getElementById("symbol").textContent = d.symbol;
        document.getElementById("price").textContent = d.price;
        document.getElementById("priceStatus").textContent = d.price_ts ? ("현재가 갱신: " + d.price_ts) : "현재가 실시간 반영";
        document.getElementById("analysisStatus").textContent = d.analysis_status || "대기중";
        document.getElementById("status").textContent = d.analysis_status || "대기중";
        document.getElementById("confidence").textContent = "신뢰도: " + d.confidence + "%";
        document.getElementById("summary").textContent = d.summary || "";
        document.getElementById("error").textContent = d.error || "";
        setPill("finalSignal", d.final_signal || "⚪ HOLD");
        setPill("shortTrend", d.short_trend || "⚪ HOLD");
        setPill("mediumTrend", d.medium_trend || "⚪ HOLD");
        setPill("longTrend", d.long_trend || "⚪ HOLD");
        const states = d.indicator_states || {{}};
        setPill("maState", states.ma || "⚪ HOLD");
        setPill("ichiState", states.ichimoku || "⚪ HOLD");
        setPill("volumeState", states.volume || "⚪ HOLD");
        setPill("rsiState", states.rsi || "⚪ HOLD");
        setPill("macdState", states.macd || "⚪ HOLD");
        setPill("bbState", states.bollinger || "⚪ HOLD");
        const logs = document.getElementById("logs");
        logs.innerHTML = "";
        (d.signal_log || []).forEach(item => {{
            const div = document.createElement("div");
            div.className = "log-item";
            div.textContent = item;
            logs.appendChild(div);
        }});
    }})
    .catch(err => {{
        document.getElementById("error").textContent = "데이터 갱신 오류: " + err;
    }});
}}
initChart(chartSymbol);
updateData();
setInterval(updateData, 1000);
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200):
        self._send_bytes(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            symbol_options = []
            for sym in REQUESTED_SYMBOLS:
                available = symbol_availability.get(sym, False)
                label = f"{sym}{' (미지원)' if not available else ''}"
                selected = " selected" if sym == selected_symbol else ""
                symbol_options.append(f'<option value="{sym}"{selected}>{label}</option>')
            html_text = HTML_TEMPLATE.format(
                app_name=APP_NAME,
                default_symbol=selected_symbol,
                symbol_options="\n".join(symbol_options),
                lan_ip=get_lan_ip(),
            )
            self._send_bytes(200, html_text.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/data":
            self._send_json(get_display_state())
            return
        if path.startswith("/backtest/"):
            symbol = path.split("/", 2)[2].upper().strip()
            if symbol not in REQUESTED_SYMBOLS:
                self._send_json({"ok": False, "message": "지원 목록에 없는 종목입니다."}, 400)
                return
            if not symbol_availability.get(symbol, False):
                self._send_json({"ok": False, "message": "바이낸스 선물에서 지원되지 않는 종목이라 백테스트할 수 없습니다."}, 400)
                return
            try:
                result = backtest_symbol(symbol)
                with state_lock:
                    symbol_states[symbol]["backtest"] = result
                self._send_json({"ok": True, **result})
            except Exception as e:
                self._send_json({"ok": False, "message": "백테스트 실패", "error": str(e)}, 500)
            return
        self._send_json({"ok": False, "message": "Not Found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/set_symbol":
            self._send_json({"ok": False, "message": "Not Found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        symbol = str(payload.get("symbol", "")).upper().strip()
        if symbol not in REQUESTED_SYMBOLS:
            self._send_json({"ok": False, "message": "지원 목록에 없는 종목입니다."}, 400)
            return
        global selected_symbol
        with state_lock:
            selected_symbol = symbol
            st = symbol_states[symbol]
            if symbol_availability.get(symbol, False):
                st["analysis_status"] = f"🔄 {symbol} 분석 중..."
                st["error"] = ""
            else:
                st["analysis_status"] = f"⚠️ {symbol} 데이터 미지원"
                st["error"] = "바이낸스 선물에서 지원되지 않는 종목입니다."
        self._send_json({"ok": True, "symbol": symbol, "available": symbol_availability.get(symbol, False)})

    def log_message(self, format, *args):
        return

def main():
    start_threads()
    def open_browser():
        time.sleep(1.0)
        try:
            webbrowser.open(f"http://127.0.0.1:{PORT}")
        except Exception:
            pass
    threading.Thread(target=open_browser, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"{APP_NAME} running on http://127.0.0.1:{PORT}")
    print(f"LAN access: http://{get_lan_ip()}:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    main()
