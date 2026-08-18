import time
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from tvDatafeed import TvDatafeed, Interval
except Exception:
    TvDatafeed = None
    Interval = None


# ============================================================
# EMA ÜSTÜ TARAMA - TRADINGVIEW UYUMLU SÜRÜM
#
# GÜNCEL FİYAT:
#   TradingView Türkiye Screener "close"
#
# EMA'LAR:
#   D / W / M:
#       TradingView Screener native EMA5/21/50/200
#
#   3M / 6M / 12M:
#       TradingView aylık kapanış geçmişinden oluşturulur
#       ve eski Colab mantığıyla pandas EWM kullanılır.
#
# 5Y CAMARILLA P:
#   TradingView 12M
#
# ANA KOŞUL:
#   Fiyat 24 EMA'nın TAMAMININ üzerinde
#   ve EN YÜKSEK EMA'dan en fazla %3 yukarıda.
#
# R4 YOK.
# ============================================================

st.set_page_config(
    page_title="EMA ÜSTÜ TARAMA",
    page_icon="📈",
    layout="wide",
)

EMA_LIST = [5, 21, 50, 200]
MAX_UZAKLIK_PCT = 3.0

SCANNER_URL = "https://scanner.tradingview.com/turkey/scan"
MONTHLY_BARS = 1200
PIVOT_12M_BARS = 20

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


# ============================================================
# TRADINGVIEW SCREENER:
# BIST + GÜNCEL FİYAT + D/W/M NATIVE EMA
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_tv_scanner_data():
    cols = ["name", "close"]

    # Günlük
    for length in EMA_LIST:
        cols.append(f"EMA{length}")

    # Haftalık
    for length in EMA_LIST:
        cols.append(f"EMA{length}|1W")

    # Aylık
    for length in EMA_LIST:
        cols.append(f"EMA{length}|1M")

    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "BIST"},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "options": {"lang": "tr"},
        "markets": ["turkey"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": cols,
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 5000],
    }

    r = requests.post(
        SCANNER_URL,
        json=payload,
        timeout=40,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    js = r.json()

    rows = []

    for item in js.get("data", []):
        d = item.get("d", [])
        if not d or len(d) < len(cols):
            continue

        row = dict(zip(cols, d))
        symbol = str(row.get("name", "")).strip().upper()

        if not symbol:
            continue

        out = {
            "Hisse": symbol,
            "Güncel Fiyat": pd.to_numeric(row.get("close"), errors="coerce"),
            "Fiyat Kaynağı": "TradingView",
        }

        # D
        for length in EMA_LIST:
            out[f"D_EMA{length}"] = pd.to_numeric(
                row.get(f"EMA{length}"),
                errors="coerce",
            )

        # W
        for length in EMA_LIST:
            out[f"W_EMA{length}"] = pd.to_numeric(
                row.get(f"EMA{length}|1W"),
                errors="coerce",
            )

        # M
        for length in EMA_LIST:
            out[f"M_EMA{length}"] = pd.to_numeric(
                row.get(f"EMA{length}|1M"),
                errors="coerce",
            )

        rows.append(out)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("TradingView Screener veri döndürmedi.")

    df = (
        df.drop_duplicates(subset=["Hisse"], keep="last")
        .sort_values("Hisse")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TVDATAFEED
# ============================================================

@st.cache_resource
def get_tv_client():
    if TvDatafeed is None or Interval is None:
        raise RuntimeError(
            "tvDatafeed kurulu değil. requirements.txt kontrol edilmeli."
        )
    return TvDatafeed()


def clean_monthly_close(df):
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    close = pd.to_numeric(df["close"], errors="coerce").dropna()

    idx = pd.to_datetime(close.index, errors="coerce")
    valid = ~idx.isna()

    close = close.loc[valid]
    idx = idx[valid]

    # Aylık bar timestamp'i ayın hangi gününe verilirse verilsin,
    # takvimsel ay sonuna normalize et.
    month_end = idx.to_period("M").to_timestamp("M")

    close.index = month_end
    close = close[~close.index.duplicated(keep="last")]
    close = close.sort_index()

    return close


def get_tv_monthly_close(symbol):
    tv = get_tv_client()
    last_err = None

    for attempt in range(3):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="BIST",
                interval=Interval.in_monthly,
                n_bars=MONTHLY_BARS,
                extended_session=False,
            )

            close = clean_monthly_close(df)

            if not close.empty:
                return close

        except Exception as e:
            last_err = e
            time.sleep(0.8 + attempt)

    raise RuntimeError(
        f"{symbol}: TradingView aylık veri alınamadı. Son hata: {last_err}"
    )


# ============================================================
# 3M / 6M / 12M
# ============================================================

def resample_from_monthly(monthly_close, tf):
    s = monthly_close.copy()

    if s is None or s.empty:
        return pd.Series(dtype=float)

    if tf == "3M":
        out = s.resample("QE-DEC").last()

    elif tf == "6M":
        # Tamamlanmış Haz/Aralık barları
        out = s[s.index.month.isin([6, 12])].copy()

        # İçinde bulunulan yarıyıl henüz tamamlanmadıysa
        # mevcut aylık kapanışı o yarıyılın güncel kapanışı say.
        last_date = s.index[-1]
        half_end_month = 6 if last_date.month <= 6 else 12

        label = (
            pd.Timestamp(
                year=last_date.year,
                month=half_end_month,
                day=1,
            )
            + pd.offsets.MonthEnd(0)
        )

        already_current = (
            len(out) > 0
            and out.index[-1].year == last_date.year
            and out.index[-1].month == half_end_month
        )

        if not already_current:
            out.loc[label] = s.iloc[-1]
            out = out.sort_index()

    elif tf == "12M":
        out = s.resample("YE-DEC").last()

    else:
        raise ValueError(f"Bilinmeyen periyot: {tf}")

    return out.dropna()


def ema_last(series, length):
    if series is None or len(series) == 0:
        return np.nan

    return float(
        series.ewm(
            span=length,
            adjust=False,
        ).mean().iloc[-1]
    )


def calc_long_tf_emas(monthly_close):
    vals = {}

    for tf in ["3M", "6M", "12M"]:
        s = resample_from_monthly(monthly_close, tf)

        for length in EMA_LIST:
            vals[f"{tf}_EMA{length}"] = ema_last(s, length)

    return vals


# ============================================================
# 24 EMA KONTROLÜ
# ============================================================

def analyze_24_ema(row, long_emas):
    price = pd.to_numeric(row.get("Güncel Fiyat"), errors="coerce")

    if pd.isna(price) or price <= 0:
        return None

    ema_values = {}

    # TradingView native D/W/M
    for tf in ["D", "W", "M"]:
        for length in EMA_LIST:
            col = f"{tf}_EMA{length}"
            ema_values[col] = pd.to_numeric(
                row.get(col),
                errors="coerce",
            )

    # TradingView aylık geçmişinden 3M/6M/12M
    for tf in ["3M", "6M", "12M"]:
        for length in EMA_LIST:
            col = f"{tf}_EMA{length}"
            ema_values[col] = pd.to_numeric(
                long_emas.get(col),
                errors="coerce",
            )

    valid = {
        k: float(v)
        for k, v in ema_values.items()
        if pd.notna(v) and np.isfinite(v) and float(v) > 0
    }

    # 24'ünün de gerçekten bulunması zorunlu.
    tum_ema_var = len(valid) == 24

    if not tum_ema_var:
        return {
            "24 EMA TÜMÜ VAR": False,
            "24 EMA TÜMÜ ÜSTÜ": False,
            "EN YÜKSEK EMA MAX %3": False,
            "En Yüksek EMA Adı": "",
            "En Yüksek EMA": np.nan,
            "En Yüksek EMA Uzaklık %": np.nan,
            **ema_values,
        }

    # İkinci güvenlik kontrolü:
    # TEK BİR EMA dahi fiyata eşit/yukarıdaysa listeye GİREMEZ.
    below_or_equal = {
        name: value
        for name, value in valid.items()
        if not (price > value)
    }

    tumu_ustu = len(below_or_equal) == 0

    max_ema_name = max(valid, key=valid.get)
    max_ema = float(valid[max_ema_name])

    uzaklik_pct = (
        ((float(price) / max_ema) - 1.0) * 100.0
        if tumu_ustu and max_ema > 0
        else np.nan
    )

    max3 = bool(
        tumu_ustu
        and pd.notna(uzaklik_pct)
        and 0 <= uzaklik_pct <= MAX_UZAKLIK_PCT
    )

    result = {
        "24 EMA TÜMÜ VAR": True,
        "24 EMA TÜMÜ ÜSTÜ": tumu_ustu,
        "EN YÜKSEK EMA MAX %3": max3,
        "En Yüksek EMA Adı": max_ema_name,
        "En Yüksek EMA": max_ema,
        "En Yüksek EMA Uzaklık %": uzaklik_pct,
    }

    result.update(ema_values)

    for name, value in valid.items():
        result[f"{name}_USTUNDE"] = bool(float(price) > value)

    return result


# ============================================================
# TÜM BIST ANALİZİ
# ============================================================

def run_ema_scan(scanner_df, progress_cb=None):
    results = []
    errors = []

    total = len(scanner_df)

    for i, row in scanner_df.iterrows():
        symbol = row["Hisse"]

        try:
            monthly_close = get_tv_monthly_close(symbol)
            long_emas = calc_long_tf_emas(monthly_close)
            analysis = analyze_24_ema(row, long_emas)

            if analysis is None:
                raise RuntimeError("Güncel TradingView fiyatı yok")

            out = {
                "Hisse": symbol,
                "Güncel Fiyat": float(row["Güncel Fiyat"]),
                "Fiyat Kaynağı": "TradingView",
                # Excel timezone-aware datetime kabul etmediği için
                # saat bilgisini metin olarak saklıyoruz.
                "Son Tarama Zamanı": datetime.now(ISTANBUL_TZ).strftime("%d.%m.%Y %H:%M:%S"),
            }

            # D/W/M native değerleri
            for tf in ["D", "W", "M"]:
                for length in EMA_LIST:
                    col = f"{tf}_EMA{length}"
                    out[col] = row[col]

            out.update(analysis)

            # SON EMNİYET:
            # Geçti denilen bir hissede 24 EMA tekrar tek tek doğrulanır.
            if out["EN YÜKSEK EMA MAX %3"]:
                all_names = [
                    f"{tf}_EMA{length}"
                    for tf in ["D", "W", "M", "3M", "6M", "12M"]
                    for length in EMA_LIST
                ]

                vals = [
                    pd.to_numeric(out.get(c), errors="coerce")
                    for c in all_names
                ]

                if (
                    len(vals) != 24
                    or any(pd.isna(v) for v in vals)
                    or not all(float(out["Güncel Fiyat"]) > float(v) for v in vals)
                ):
                    out["EN YÜKSEK EMA MAX %3"] = False
                    out["24 EMA TÜMÜ ÜSTÜ"] = False

            results.append(out)

        except Exception as e:
            errors.append({
                "Aşama": "24 EMA",
                "Hisse": symbol,
                "Hata": str(e),
            })

        if progress_cb:
            progress_cb(
                (i + 1) / max(total, 1),
                f"TradingView EMA: {i + 1}/{total}",
            )

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(
            [
                "EN YÜKSEK EMA MAX %3",
                "En Yüksek EMA Uzaklık %",
                "Hisse",
            ],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return df, errors


# ============================================================
# 5Y PIVOT/P - TRADINGVIEW 12M
# ============================================================

class TVInterval12M:
    value = "12M"


def get_tv_12m(symbol, n_bars=PIVOT_12M_BARS):
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
                x.index = pd.to_datetime(
                    x.index,
                    errors="coerce",
                )
                x = x[~x.index.isna()].sort_index()

                for c in ["open", "high", "low", "close"]:
                    if c in x.columns:
                        x[c] = pd.to_numeric(
                            x[c],
                            errors="coerce",
                        )

                x = x.dropna(
                    subset=["high", "low", "close"]
                )

                return x

        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)

    raise RuntimeError(
        f"{symbol}: TradingView 12M veri alınamadı. Son hata: {last_err}"
    )


def pine_5y_p_from_tv_12m(tv12m):
    x = tv12m.copy()

    if x is None or x.empty:
        return None

    x["Year"] = x.index.year.astype(int)

    current_year = datetime.now(ISTANBUL_TZ).year

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

    expected = set(
        range(start_year, end_year + 1)
    )
    present = set(ref["Year"].tolist())

    if not expected.issubset(present):
        return None

    ref = (
        ref.sort_index()
        .drop_duplicates(
            subset=["Year"],
            keep="last",
        )
    )

    if len(ref) < 5:
        return None

    H = float(ref["high"].max())
    L = float(ref["low"].min())

    end_row = ref[
        ref["Year"] == end_year
    ]

    if end_row.empty:
        return None

    C = float(
        end_row["close"].iloc[-1]
    )

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
            annual = get_tv_12m(symbol)
            info = pine_5y_p_from_tv_12m(annual)

            if info is None:
                raise RuntimeError(
                    "Pine 5Y blok hesabı üretilemedi"
                )

            p = float(info["P"])

            p_mesafe = (
                ((price / p) - 1.0) * 100.0
                if p > 0
                else np.nan
            )

            if price < p:
                p_konum = "P ALTINDA"
            elif price > p:
                p_konum = "P ÜSTÜNDE"
            else:
                p_konum = "P ÜZERİNDE"

            rows.append({
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
            })

        except Exception as e:
            errors.append({
                "Aşama": "TradingView 5Y P",
                "Hisse": symbol,
                "Hata": str(e),
            })

        if progress_cb:
            progress_cb(
                (i + 1) / max(total, 1),
                f"5Y Pivot/P: {i + 1}/{total}",
            )

    p_df = pd.DataFrame(rows)

    if p_df.empty:
        return df.copy(), errors

    final = df.merge(
        p_df,
        on="Hisse",
        how="left",
    )

    if len(final) != len(df):
        raise RuntimeError(
            "P eklenirken hisse sayısı değişti."
        )

    return final, errors


# ============================================================
# EXCEL
# ============================================================

def make_excel(df, errors):
    out = df.copy()

    # Excel timezone-aware datetime değerlerini kabul etmez.
    # Herhangi bir datetime sütunu kaldıysa güvenli şekilde timezone bilgisini kaldır.
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            try:
                if getattr(out[c].dt, "tz", None) is not None:
                    out[c] = out[c].dt.tz_localize(None)
            except Exception:
                pass

    if "Güncel Fiyat" in out.columns:
        out["Güncel Fiyat"] = pd.to_numeric(
            out["Güncel Fiyat"],
            errors="coerce",
        ).round(2)

    price_cols_4 = [
        "En Yüksek EMA",
        "5Y Camarilla P",
        "5Y High (TV 12M)",
        "5Y Low (TV 12M)",
        "5Y Close (TV 12M)",
    ]

    for c in price_cols_4:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce",
            ).round(4)

    pct_cols = [
        "En Yüksek EMA Uzaklık %",
        "5Y P Mesafe %",
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce",
            ).round(2)

    ema_cols = [
        c for c in out.columns
        if "_EMA" in c
        and "USTUNDE" not in c
        and c not in [
            "En Yüksek EMA Adı",
            "En Yüksek EMA Uzaklık %",
        ]
    ]

    for c in ema_cols:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        ).round(4)

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

        out.to_excel(
            writer,
            sheet_name="SONUCLAR",
            index=False,
        )

        err_df = pd.DataFrame(errors)

        if err_df.empty:
            err_df = pd.DataFrame(
                columns=[
                    "Aşama",
                    "Hisse",
                    "Hata",
                ]
            )

        err_df.to_excel(
            writer,
            sheet_name="HATALAR",
            index=False,
        )

        ws = writer.book["SONUCLAR"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        headers = {
            cell.value: cell.column
            for cell in ws[1]
        }

        if "Güncel Fiyat" in headers:
            col = headers["Güncel Fiyat"]
            for r in range(
                2,
                ws.max_row + 1,
            ):
                ws.cell(
                    r,
                    col,
                ).number_format = "0.00"

        for name in price_cols_4:
            if name in headers:
                col = headers[name]
                for r in range(
                    2,
                    ws.max_row + 1,
                ):
                    ws.cell(
                        r,
                        col,
                    ).number_format = "0.0000"

        for name in pct_cols:
            if name in headers:
                col = headers[name]
                for r in range(
                    2,
                    ws.max_row + 1,
                ):
                    ws.cell(
                        r,
                        col,
                    ).number_format = "0.00"

        for name in ema_cols:
            if name in headers:
                col = headers[name]
                for r in range(
                    2,
                    ws.max_row + 1,
                ):
                    ws.cell(
                        r,
                        col,
                    ).number_format = "0.0000"

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# ARAYÜZ
# ============================================================

st.title("EMA ÜSTÜ TARAMA")

st.caption(
    "BIST — 24 EMA üstü + en yüksek EMA'dan MAX %3 + 5Y Pivot/P"
)

st.info(
    "Güncel Fiyat ve D/W/M EMA: TradingView Screener. "
    "3M/6M/12M EMA: TradingView aylık geçmişi. "
    "5Y Pivot/P: TradingView 12M. "
    "Yahoo Finance kullanılmaz."
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
        scanner_df = get_tv_scanner_data()

        progress = st.progress(
            0.0,
            text=(
                f"TradingView BIST listesi: "
                f"{len(scanner_df)} hisse"
            ),
        )

        df_all, ema_errors = run_ema_scan(
            scanner_df,
            lambda p, t: progress.progress(
                min(p, 1.0),
                text=t,
            ),
        )

        passed = (
            df_all[
                df_all[
                    "EN YÜKSEK EMA MAX %3"
                ] == True
            ]
            .copy()
            .reset_index(drop=True)
        )

        # SANFM özel güvenlik notu:
        # SANFM ancak gerçekten bütün 24 EMA üstündeyse listede olabilir.
        if "SANFM" in passed["Hisse"].tolist():
            san = passed[
                passed["Hisse"] == "SANFM"
            ].iloc[0]

            all_ema_cols = [
                f"{tf}_EMA{length}"
                for tf in [
                    "D",
                    "W",
                    "M",
                    "3M",
                    "6M",
                    "12M",
                ]
                for length in EMA_LIST
            ]

            san_price = float(
                san["Güncel Fiyat"]
            )

            san_ok = all(
                pd.notna(san[c])
                and san_price > float(san[c])
                for c in all_ema_cols
            )

            if not san_ok:
                passed = passed[
                    passed["Hisse"] != "SANFM"
                ].reset_index(drop=True)

                ema_errors.append({
                    "Aşama": "SANFM Güvenlik",
                    "Hisse": "SANFM",
                    "Hata": (
                        "24 EMA doğrulamasını geçemediği "
                        "için final listeden çıkarıldı."
                    ),
                })

        if passed.empty:
            progress.empty()
            st.session_state["scan_df"] = passed
            st.session_state["errors"] = ema_errors
            st.warning(
                "EMA filtresinden geçen hisse bulunamadı."
            )

        else:
            progress.progress(
                0.0,
                text=(
                    f"EMA'dan geçen {len(passed)} "
                    f"hisseye 5Y Pivot/P hesaplanıyor..."
                ),
            )

            final, p_errors = add_5y_p(
                passed,
                lambda p, t: progress.progress(
                    min(p, 1.0),
                    text=t,
                ),
            )

            # Orijinal Colab sırası:
            # negatif P mesafeden pozitife.
            if "5Y P Mesafe %" in final.columns:
                final = final.sort_values(
                    [
                        "5Y P Mesafe %",
                        "Hisse",
                    ],
                    ascending=[True, True],
                    na_position="last",
                ).reset_index(drop=True)

            progress.empty()

            st.session_state["scan_df"] = final
            st.session_state["errors"] = (
                ema_errors + p_errors
            )

    except Exception as e:
        st.error(
            f"Tarama hatası: {e}"
        )


df = st.session_state.get("scan_df")
errors = st.session_state.get(
    "errors",
    [],
)


if df is not None:

    if df.empty:
        st.warning("Sonuç bulunamadı.")

    else:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "EMA Filtresinden Geçen",
            len(df),
        )

        c2.metric(
            "MAX EMA Uzaklık",
            "%3",
        )

        c3.metric(
            "Pivot",
            "5Y P",
        )

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
            "Fiyat Kaynağı",
            "5Y Camarilla P",
            "5Y P Mesafe %",
            "5Y P Konum",
            "En Yüksek EMA Adı",
            "En Yüksek EMA",
            "En Yüksek EMA Uzaklık %",
            "Son Tarama Zamanı",
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

        show = show[
            front_cols + other_cols
        ]

        display_df = show.copy()

        if "Güncel Fiyat" in display_df.columns:
            display_df[
                "Güncel Fiyat"
            ] = pd.to_numeric(
                display_df[
                    "Güncel Fiyat"
                ],
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
                "Güncel Fiyat":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "5Y Camarilla P":
                    st.column_config.NumberColumn(
                        format="%.4f"
                    ),
                "5Y P Mesafe %":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "En Yüksek EMA":
                    st.column_config.NumberColumn(
                        format="%.4f"
                    ),
                "En Yüksek EMA Uzaklık %":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                **{
                    c:
                    st.column_config.NumberColumn(
                        format="%.4f"
                    )
                    for c in ema_value_cols
                },
            },
        )

        excel_bytes = make_excel(
            show,
            errors,
        )

        file_name = (
            "EMA_USTU_TARAMA_TV_"
            + datetime.now(
                ISTANBUL_TZ
            ).strftime(
                "%Y%m%d_%H%M"
            )
            + ".xlsx"
        )

        st.download_button(
            "SONUÇLARI EXCEL İNDİR",
            data=excel_bytes,
            file_name=file_name,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

        st.caption(
            "Bir hisse final listeye girmeden önce "
            "24 EMA'nın tamamı ikinci kez tek tek doğrulanır. "
            "P Mesafe % = ((Güncel Fiyat / 5Y P) - 1) × 100."
        )


if errors:
    with st.expander(
        f"Hatalar ({len(errors)})"
    ):
        st.dataframe(
            pd.DataFrame(errors),
            use_container_width=True,
            hide_index=True,
        )
