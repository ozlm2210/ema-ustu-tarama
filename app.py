import time
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from tvDatafeed import TvDatafeed
except Exception:
    TvDatafeed = None


# ============================================================
# EMA ÜSTÜ TARAMA
# BIST 24 EMA / TÜM PERİYOTLAR / MAX %3
# EMA: Yahoo Finance ADJ KAPALI
# 5Y Pivot (P): TradingView 12M
# ============================================================

st.set_page_config(
    page_title="EMA ÜSTÜ TARAMA",
    page_icon="📈",
    layout="wide",
)

EMA_LIST = [5, 21, 50, 200]
MAX_UZAKLIK_PCT = 3.0
PERIOD = "max"
INTERVAL = "1d"
CHUNK_SIZE = 40
RETRY = 2
SLEEP_BETWEEN_CHUNKS = 0.4


# ============================================================
# BIST LİSTESİ
# ============================================================

@st.cache_data(ttl=3600)
def get_bist_symbols():
    url = "https://scanner.tradingview.com/turkey/scan"

    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "BIST"},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "options": {"lang": "tr"},
        "markets": ["turkey"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 5000],
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    js = r.json()

    symbols = []
    for row in js.get("data", []):
        d = row.get("d", [])
        if d and d[0]:
            s = str(d[0]).strip().upper()
            if s and s not in symbols:
                symbols.append(s)

    return sorted(symbols)


# ============================================================
# EMA YARDIMCILARI
# ============================================================

def normalize_downloaded_close(downloaded, yahoo_symbol):
    if downloaded is None or downloaded.empty:
        return pd.Series(dtype=float)

    df = downloaded.copy()

    if isinstance(df.columns, pd.MultiIndex):
        close = None

        if ("Close", yahoo_symbol) in df.columns:
            close = df[("Close", yahoo_symbol)]
        else:
            try:
                close_block = df["Close"]
                if isinstance(close_block, pd.DataFrame):
                    if yahoo_symbol in close_block.columns:
                        close = close_block[yahoo_symbol]
                    elif close_block.shape[1] == 1:
                        close = close_block.iloc[:, 0]
                else:
                    close = close_block
            except Exception:
                close = None

        if close is None:
            return pd.Series(dtype=float)

        return pd.to_numeric(close, errors="coerce").dropna()

    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna()


def clean_close(close):
    close = pd.Series(close).copy()
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close[~close.index.isna()]
    close = close[~close.index.duplicated(keep="last")]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0].sort_index()
    return close


def resample_close(close, tf):
    close = clean_close(close)

    if close.empty:
        return close

    if tf == "D":
        return close

    if tf == "W":
        out = close.resample("W-FRI").last()

    elif tf == "M":
        out = close.resample("ME").last()

    elif tf == "3M":
        out = close.resample("QE-DEC").last()

    elif tf == "6M":
        m = close.resample("ME").last().dropna()
        out = m[m.index.month.isin([6, 12])].copy()

        last_date = close.index[-1]
        current_half_end_month = 6 if last_date.month <= 6 else 12
        current_half_year = last_date.year

        already_has_current = (
            len(out) > 0
            and out.index[-1].year == current_half_year
            and out.index[-1].month == current_half_end_month
        )

        if not already_has_current:
            current_label = (
                pd.Timestamp(
                    year=current_half_year,
                    month=current_half_end_month,
                    day=1,
                )
                + pd.offsets.MonthEnd(0)
            )
            out.loc[current_label] = close.iloc[-1]
            out = out.sort_index()

    elif tf == "12M":
        out = close.resample("YE-DEC").last()

    else:
        raise ValueError(f"Bilinmeyen periyot: {tf}")

    return out.dropna()


def ema_last(series, length):
    if series is None or len(series) == 0:
        return np.nan
    return float(series.ewm(span=length, adjust=False).mean().iloc[-1])


def analyze_symbol(close, symbol):
    close = clean_close(close)

    if close.empty:
        return None

    price = float(close.iloc[-1])
    last_date = close.index[-1]

    tf_map = {
        "D": resample_close(close, "D"),
        "W": resample_close(close, "W"),
        "M": resample_close(close, "M"),
        "3M": resample_close(close, "3M"),
        "6M": resample_close(close, "6M"),
        "12M": resample_close(close, "12M"),
    }

    ema_values = {}
    base = {
        "Hisse": symbol,
        "Güncel Fiyat": price,
        "Son Veri Tarihi": last_date.date(),
    }

    for tf, s in tf_map.items():
        for length in EMA_LIST:
            col = f"{tf}_EMA{length}"
            val = ema_last(s, length)
            ema_values[col] = val
            base[col] = val
            base[f"{col}_USTUNDE"] = bool(pd.notna(val) and price > val)

    valid_emas = {k: v for k, v in ema_values.items() if pd.notna(v)}
    tum_ema_var = len(valid_emas) == 24

    if tum_ema_var:
        max_ema_name = max(valid_emas, key=valid_emas.get)
        max_ema = float(valid_emas[max_ema_name])
        tumu_ustu = all(price > v for v in valid_emas.values())

        uzaklik_pct = (
            ((price / max_ema) - 1.0) * 100.0
            if max_ema > 0 and price > max_ema
            else np.nan
        )

        max3 = bool(
            tumu_ustu
            and pd.notna(uzaklik_pct)
            and 0 <= uzaklik_pct <= MAX_UZAKLIK_PCT
        )
    else:
        max_ema_name = ""
        max_ema = np.nan
        tumu_ustu = False
        uzaklik_pct = np.nan
        max3 = False

    summary = {
        "Hisse": symbol,
        "Güncel Fiyat": price,
        "Son Veri Tarihi": last_date.date(),
        "24 EMA TÜMÜ VAR": tum_ema_var,
        "24 EMA TÜMÜ ÜSTÜ": tumu_ustu,
        "EN YÜKSEK EMA MAX %3": max3,
        "En Yüksek EMA Adı": max_ema_name,
        "En Yüksek EMA": max_ema,
        "En Yüksek EMA Uzaklık %": uzaklik_pct,
    }

    for tf in ["D", "W", "M", "3M", "6M", "12M"]:
        for length in EMA_LIST:
            col = f"{tf}_EMA{length}"
            summary[col] = base[col]
            summary[f"{col}_USTUNDE"] = base[f"{col}_USTUNDE"]

    return summary


def download_mode(symbols, progress_cb=None):
    results = []
    errors = []

    yahoo_symbols = [f"{s}.IS" for s in symbols]
    total = len(yahoo_symbols)

    for start in range(0, total, CHUNK_SIZE):
        chunk = yahoo_symbols[start : start + CHUNK_SIZE]

        data = None
        last_err = ""
        ok = False

        for attempt in range(RETRY + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=PERIOD,
                    interval=INTERVAL,
                    auto_adjust=False,  # ADJ KAPALI
                    actions=False,
                    group_by="column",
                    threads=True,
                    progress=False,
                    timeout=30,
                )
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0 + attempt)

        if not ok:
            for ys in chunk:
                errors.append(
                    {
                        "Aşama": "EMA",
                        "Hisse": ys.replace(".IS", ""),
                        "Hata": f"İndirme başarısız: {last_err}",
                    }
                )
            continue

        for ys in chunk:
            symbol = ys.replace(".IS", "")

            try:
                close = normalize_downloaded_close(data, ys)

                if close.empty:
                    errors.append(
                        {
                            "Aşama": "EMA",
                            "Hisse": symbol,
                            "Hata": "Close verisi yok",
                        }
                    )
                    continue

                row = analyze_symbol(close, symbol)

                if row is None:
                    errors.append(
                        {
                            "Aşama": "EMA",
                            "Hisse": symbol,
                            "Hata": "Analiz için veri yok",
                        }
                    )
                else:
                    results.append(row)

            except Exception as e:
                errors.append(
                    {
                        "Aşama": "EMA",
                        "Hisse": symbol,
                        "Hata": str(e),
                    }
                )

        done = min(start + CHUNK_SIZE, total)

        if progress_cb:
            progress_cb(done / max(total, 1), f"EMA: {done}/{total}")

        time.sleep(SLEEP_BETWEEN_CHUNKS)

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(
            by=[
                "EN YÜKSEK EMA MAX %3",
                "En Yüksek EMA Uzaklık %",
                "Hisse",
            ],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return df, errors


# ============================================================
# TRADINGVIEW 12M / 5Y P
# ============================================================

class TVInterval12M:
    value = "12M"


@st.cache_resource
def get_tv_client():
    if TvDatafeed is None:
        raise RuntimeError(
            "tvDatafeed kurulu değil. requirements.txt kontrol edilmeli."
        )
    return TvDatafeed()


def get_tv_12m(symbol, n_bars=20):
    tv = get_tv_client()
    last_err = None

    for attempt in range(3):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="BIST",
                interval=TVInterval12M(),
                n_bars=n_bars,
                extended_session=False,
            )

            if df is not None and not df.empty:
                x = df.copy()
                x.index = pd.to_datetime(x.index, errors="coerce")
                x = x[~x.index.isna()].sort_index()

                for c in ["open", "high", "low", "close"]:
                    if c in x.columns:
                        x[c] = pd.to_numeric(x[c], errors="coerce")

                x = x.dropna(subset=["high", "low", "close"])
                return x

        except Exception as e:
            last_err = e
            time.sleep(1.2 + attempt)

    raise RuntimeError(
        f"{symbol}: TradingView 12M veri alınamadı. Son hata: {last_err}"
    )


def pine_5y_p_from_tv_12m(tv12m):
    x = tv12m.copy()

    if x is None or x.empty:
        return None

    x["Year"] = x.index.year.astype(int)
    current_year = datetime.now(ZoneInfo("Europe/Istanbul")).year

    boundary_candidates = sorted(
        y for y in x["Year"].unique()
        if y % 5 == 0 and y < current_year
    )

    if not boundary_candidates:
        return None

    boundary_year = boundary_candidates[-1]
    start_year = boundary_year - 5
    end_year = boundary_year - 1

    ref = x[
        (x["Year"] >= start_year)
        & (x["Year"] <= end_year)
    ].copy()

    expected = set(range(start_year, end_year + 1))
    present = set(ref["Year"].tolist())

    if not expected.issubset(present):
        return None

    ref = (
        ref.sort_index()
        .drop_duplicates(subset=["Year"], keep="last")
    )

    if len(ref) < 5:
        return None

    H = float(ref["high"].max())
    L = float(ref["low"].min())

    end_row = ref[ref["Year"] == end_year]
    if end_row.empty:
        return None

    C = float(end_row["close"].iloc[-1])
    P = (H + L + C) / 3.0

    return {
        "BoundaryYear": boundary_year,
        "StartYear": start_year,
        "EndYear": end_year,
        "High": H,
        "Low": L,
        "Close": C,
        "P": P,
    }


def add_5y_p(df, progress_cb=None):
    rows = []
    errors = []
    total = len(df)

    for i, row in df.reset_index(drop=True).iterrows():
        symbol = row["Hisse"]
        price = float(row["Güncel Fiyat"])

        try:
            annual = get_tv_12m(symbol, n_bars=20)
            info = pine_5y_p_from_tv_12m(annual)

            if info is None:
                raise RuntimeError("Pine 5Y blok hesabı üretilemedi")

            p = float(info["P"])
            p_mesafe = ((price / p) - 1.0) * 100.0

            if price < p:
                p_konum = "P ALTINDA"
            elif price > p:
                p_konum = "P ÜSTÜNDE"
            else:
                p_konum = "P ÜZERİNDE"

            rows.append(
                {
                    "Hisse": symbol,
                    "5Y Camarilla P": p,
                    "5Y P Mesafe %": p_mesafe,
                    "5Y P Konum": p_konum,
                    "5Y Blok Sınır Yılı": info["BoundaryYear"],
                    "5Y Başlangıç Yılı": info["StartYear"],
                    "5Y Bitiş Yılı": info["EndYear"],
                    "5Y High (TV 12M)": info["High"],
                    "5Y Low (TV 12M)": info["Low"],
                    "5Y Close (TV 12M)": info["Close"],
                }
            )

        except Exception as e:
            errors.append(
                {
                    "Aşama": "TradingView 5Y P",
                    "Hisse": symbol,
                    "Hata": str(e),
                }
            )

        if progress_cb:
            progress_cb(
                (i + 1) / max(total, 1),
                f"5Y Pivot/P: {i + 1}/{total}",
            )

    p_df = pd.DataFrame(rows)

    if p_df.empty:
        return df.copy(), errors

    final = df.merge(p_df, on="Hisse", how="left")

    if len(final) != len(df):
        raise RuntimeError("P eklenirken hisse sayısı değişti.")

    return final, errors


# ============================================================
# EXCEL
# ============================================================

def make_excel(df, errors):
    out = df.copy()

    # Görsel biçim
    if "Güncel Fiyat" in out.columns:
        out["Güncel Fiyat"] = pd.to_numeric(
            out["Güncel Fiyat"], errors="coerce"
        ).round(2)

    for c in ["En Yüksek EMA", "5Y Camarilla P",
              "5Y High (TV 12M)", "5Y Low (TV 12M)",
              "5Y Close (TV 12M)"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    for c in ["En Yüksek EMA Uzaklık %", "5Y P Mesafe %"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    ema_cols = [
        c for c in out.columns
        if "_EMA" in c
        and "USTUNDE" not in c
        and c not in ["En Yüksek EMA Adı", "En Yüksek EMA Uzaklık %"]
    ]

    for c in ema_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="SONUCLAR", index=False)

        err_df = pd.DataFrame(errors)
        if err_df.empty:
            err_df = pd.DataFrame(
                columns=["Aşama", "Hisse", "Hata"]
            )
        err_df.to_excel(writer, sheet_name="HATALAR", index=False)

        ws = writer.book["SONUCLAR"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        headers = {cell.value: cell.column for cell in ws[1]}

        if "Güncel Fiyat" in headers:
            col = headers["Güncel Fiyat"]
            for r in range(2, ws.max_row + 1):
                ws.cell(r, col).number_format = "0.00"

        for name in ["En Yüksek EMA", "5Y Camarilla P",
                     "5Y High (TV 12M)", "5Y Low (TV 12M)",
                     "5Y Close (TV 12M)"]:
            if name in headers:
                col = headers[name]
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, col).number_format = "0.0000"

        for name in ["En Yüksek EMA Uzaklık %", "5Y P Mesafe %"]:
            if name in headers:
                col = headers[name]
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, col).number_format = "0.00"

        for name in ema_cols:
            if name in headers:
                col = headers[name]
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, col).number_format = "0.0000"

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# ARAYÜZ
# ============================================================

st.title("EMA ÜSTÜ TARAMA")
st.caption(
    "BIST Çoklu Periyot EMA Tarayıcı — 24 EMA üstü + en yüksek EMA'dan MAX %3 + 5Y Pivot/P uzaklığı"
)

st.info(
    "EMA veri kaynağı: Yahoo Finance (ADJ KAPALI). "
    "5Y Pivot/P veri kaynağı: TradingView 12M."
)

if "scan_df" not in st.session_state:
    st.session_state["scan_df"] = None
if "errors" not in st.session_state:
    st.session_state["errors"] = []

if st.button(
    "TARAMAYI BAŞLAT",
    type="primary",
    use_container_width=True,
):
    try:
        symbols = get_bist_symbols()

        progress = st.progress(
            0.0,
            text=f"BIST listesi hazır: {len(symbols)} hisse",
        )

        df_all, ema_errors = download_mode(
            symbols,
            lambda p, t: progress.progress(min(p, 1.0), text=t),
        )

        passed = (
            df_all[
                df_all["EN YÜKSEK EMA MAX %3"] == True
            ]
            .copy()
            .reset_index(drop=True)
        )

        if passed.empty:
            progress.empty()
            st.session_state["scan_df"] = passed
            st.session_state["errors"] = ema_errors
            st.warning("EMA filtresinden geçen hisse bulunamadı.")
        else:
            progress.progress(
                0.0,
                text=f"EMA'dan geçen {len(passed)} hisseye 5Y Pivot/P hesaplanıyor...",
            )

            final, p_errors = add_5y_p(
                passed,
                lambda p, t: progress.progress(min(p, 1.0), text=t),
            )

            # Orijinal Colab mantığı: 5Y P Mesafe % negatiften pozitife.
            if "5Y P Mesafe %" in final.columns:
                final = final.sort_values(
                    ["5Y P Mesafe %", "Hisse"],
                    ascending=[True, True],
                    na_position="last",
                ).reset_index(drop=True)

            progress.empty()

            st.session_state["scan_df"] = final
            st.session_state["errors"] = ema_errors + p_errors

    except Exception as e:
        st.error(f"Tarama hatası: {e}")


df = st.session_state.get("scan_df")
errors = st.session_state.get("errors", [])

if df is not None:
    if df.empty:
        st.warning("Sonuç bulunamadı.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("EMA Filtresinden Geçen", len(df))
        c2.metric("MAX EMA Uzaklık", "%3")
        c3.metric("Pivot", "5Y P")

        st.subheader("Sonuçlar")

        q = st.text_input(
            "Hisse ara",
            placeholder="Örn: THYAO",
        )

        show = df.copy()

        if q:
            show = show[
                show["Hisse"].str.contains(
                    q.upper(),
                    na=False,
                )
            ].copy()

        front_cols = [
            "Hisse",
            "Güncel Fiyat",
            "5Y Camarilla P",
            "5Y P Mesafe %",
            "5Y P Konum",
            "En Yüksek EMA Adı",
            "En Yüksek EMA",
            "En Yüksek EMA Uzaklık %",
            "Son Veri Tarihi",
            "5Y Blok Sınır Yılı",
            "5Y Başlangıç Yılı",
            "5Y Bitiş Yılı",
            "5Y High (TV 12M)",
            "5Y Low (TV 12M)",
            "5Y Close (TV 12M)",
        ]

        front_cols = [
            c for c in front_cols
            if c in show.columns
        ]

        other_cols = [
            c for c in show.columns
            if c not in front_cols
        ]

        show = show[front_cols + other_cols]

        display_df = show.copy()

        if "Güncel Fiyat" in display_df.columns:
            display_df["Güncel Fiyat"] = pd.to_numeric(
                display_df["Güncel Fiyat"],
                errors="coerce",
            ).round(2)

        for c in [
            "5Y Camarilla P",
            "En Yüksek EMA",
            "5Y High (TV 12M)",
            "5Y Low (TV 12M)",
            "5Y Close (TV 12M)",
        ]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(
                    display_df[c],
                    errors="coerce",
                ).round(4)

        for c in [
            "5Y P Mesafe %",
            "En Yüksek EMA Uzaklık %",
        ]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(
                    display_df[c],
                    errors="coerce",
                ).round(2)

        ema_value_cols = [
            c for c in display_df.columns
            if "_EMA" in c
            and "USTUNDE" not in c
            and c not in [
                "En Yüksek EMA Adı",
                "En Yüksek EMA Uzaklık %",
            ]
        ]

        for c in ema_value_cols:
            display_df[c] = pd.to_numeric(
                display_df[c],
                errors="coerce",
            ).round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Güncel Fiyat": st.column_config.NumberColumn(format="%.2f"),
                "5Y Camarilla P": st.column_config.NumberColumn(format="%.4f"),
                "5Y P Mesafe %": st.column_config.NumberColumn(format="%.2f"),
                "En Yüksek EMA": st.column_config.NumberColumn(format="%.4f"),
                "En Yüksek EMA Uzaklık %": st.column_config.NumberColumn(format="%.2f"),
                **{
                    c: st.column_config.NumberColumn(format="%.4f")
                    for c in ema_value_cols
                },
            },
        )

        excel_bytes = make_excel(show, errors)
        file_name = (
            "EMA_USTU_TARAMA_"
            + datetime.now(
                ZoneInfo("Europe/Istanbul")
            ).strftime("%Y%m%d_%H%M")
            + ".xlsx"
        )

        st.download_button(
            "SONUÇLARI EXCEL İNDİR",
            data=excel_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.caption(
            "P Mesafe % = ((Güncel Fiyat / 5Y P) - 1) × 100. "
            "Negatif = P altında, pozitif = P üstünde."
        )

if errors:
    with st.expander(f"Hatalar ({len(errors)})"):
        st.dataframe(
            pd.DataFrame(errors),
            use_container_width=True,
            hide_index=True,
        )
