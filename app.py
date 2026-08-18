import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO

try:
    from tvDatafeed import TvDatafeed
except Exception:
    TvDatafeed = None

st.set_page_config(page_title='EMA ÜSTÜ TARAMA', page_icon='📈', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.big-title {font-size: 2.05rem; font-weight: 800; margin-bottom: .15rem;}
.sub-title {color: #888; margin-bottom: 1.1rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 14px;}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="big-title">EMA ÜSTÜ TARAMA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">BIST Çoklu Periyot EMA Tarayıcı</div>', unsafe_allow_html=True)

EMA_LIST = [5, 21, 50, 200]
TF_LIST = ['D', 'W', 'M', '3M', '6M', '12M']
MAX_UZAKLIK_PCT = 3.0
PERIOD = 'max'
INTERVAL = '1d'
CHUNK_SIZE = 40


class TVInterval12M:
    """tvDatafeed için doğrudan TradingView 12M intervali."""
    value = "12M"


@st.cache_resource
def get_tv_client():
    if TvDatafeed is None:
        raise RuntimeError(
            "tvDatafeed kurulu değil. requirements.txt dosyasına "
            "git+https://github.com/rongardF/tvdatafeed.git eklenmeli."
        )
    return TvDatafeed()


def get_tv_12m(symbol, n_bars=20):
    """TradingView BIST 12M OHLC verisini alır."""
    tv = get_tv_client()
    last_err = None

    for attempt in range(3):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="BIST",
                interval=TVInterval12M(),
                n_bars=n_bars,
                extended_session=False
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


def pine_5y_levels_from_tv_12m(tv12m):
    """
    TradingView/Pine 5Y blok mantığı.
    2026 için referans blok 2020-2024:
      H = max(2020..2024 High)
      L = min(2020..2024 Low)
      C = 2024 Close

      P  = (H + L + C) / 3
      R4 = C + (H - L) * 1.1 / 2
    """
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

    ref = ref.sort_index().drop_duplicates(subset=["Year"], keep="last")
    if len(ref) < 5:
        return None

    H = float(ref["high"].max())
    L = float(ref["low"].min())

    end_row = ref[ref["Year"] == end_year]
    if end_row.empty:
        return None

    C = float(end_row["close"].iloc[-1])
    P = (H + L + C) / 3.0
    R4 = C + (H - L) * 1.1 / 2.0

    return {
        "BoundaryYear": boundary_year,
        "StartYear": start_year,
        "EndYear": end_year,
        "High": H,
        "Low": L,
        "Close": C,
        "P": P,
        "R4": R4,
    }


def level_distance_pct(price, level):
    if pd.isna(price) or pd.isna(level) or level == 0:
        return np.nan
    return ((float(price) / float(level)) - 1.0) * 100.0


def level_position(price, level, level_name):
    if pd.isna(price) or pd.isna(level):
        return "HESAPLANAMADI"
    if price < level:
        return f"{level_name} ALTINDA"
    if price > level:
        return f"{level_name} ÜSTÜNDE"
    return f"{level_name} ÜZERİNDE"


def add_5y_p_r4(df, progress_cb=None):
    """
    EMA filtresinden geçen hisselere yalnızca bilgi amaçlı
    5Y P/R4 ekler. Hisse listesini değiştirmez.
    """
    if df is None or df.empty:
        return df.copy(), []

    out = df.copy()
    rows = []
    errors = []
    total = len(out)

    for i, row in out.iterrows():
        symbol = row["Hisse"]
        price = row["Güncel Fiyat"]

        try:
            annual = get_tv_12m(symbol, n_bars=20)
            info = pine_5y_levels_from_tv_12m(annual)

            if info is None:
                raise RuntimeError("5Y P/R4 blok hesabı üretilemedi")

            rows.append({
                "Hisse": symbol,
                "5Y Camarilla P": info["P"],
                "5Y P Uzaklık %": level_distance_pct(price, info["P"]),
                "5Y P Konum": level_position(price, info["P"], "P"),
                "5Y Camarilla R4": info["R4"],
                "5Y R4 Uzaklık %": level_distance_pct(price, info["R4"]),
                "5Y R4 Konum": level_position(price, info["R4"], "R4"),
                "5Y Başlangıç Yılı": info["StartYear"],
                "5Y Bitiş Yılı": info["EndYear"],
                "5Y High (TV 12M)": info["High"],
                "5Y Low (TV 12M)": info["Low"],
                "5Y Close (TV 12M)": info["Close"],
            })

        except Exception as e:
            errors.append({
                "Aşama": "TradingView 5Y P/R4",
                "Hisse": symbol,
                "Hata": str(e)
            })

        if progress_cb:
            progress_cb((i + 1) / max(total, 1), f"5Y P/R4: {i + 1}/{total}")

    levels = pd.DataFrame(rows)
    if levels.empty:
        return out, errors

    merged = out.merge(levels, on="Hisse", how="left")

    if len(merged) != len(out) or set(merged["Hisse"]) != set(out["Hisse"]):
        raise RuntimeError("P/R4 eklenirken EMA hisse listesi değişti.")

    return merged, errors


def get_bist_symbols():
    url = 'https://scanner.tradingview.com/turkey/scan'
    payload = {
        'filter': [
            {'left': 'exchange', 'operation': 'equal', 'right': 'BIST'},
            {'left': 'type', 'operation': 'equal', 'right': 'stock'}
        ],
        'options': {'lang': 'tr'},
        'markets': ['turkey'],
        'symbols': {'query': {'types': []}, 'tickers': []},
        'columns': ['name'],
        'sort': {'sortBy': 'name', 'sortOrder': 'asc'},
        'range': [0, 5000]
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json().get('data', [])
    out = []
    for row in data:
        d = row.get('d', [])
        if d and d[0]:
            s = str(d[0]).strip().upper()
            if s and s not in out:
                out.append(s)
    return sorted(out)


def normalize_downloaded_close(downloaded, yahoo_symbol):
    if downloaded is None or downloaded.empty:
        return pd.Series(dtype=float)
    df = downloaded.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', yahoo_symbol) in df.columns:
            close = df[('Close', yahoo_symbol)]
        else:
            try:
                block = df['Close']
                if isinstance(block, pd.DataFrame):
                    if yahoo_symbol in block.columns:
                        close = block[yahoo_symbol]
                    elif block.shape[1] == 1:
                        close = block.iloc[:, 0]
                    else:
                        return pd.Series(dtype=float)
                else:
                    close = block
            except Exception:
                return pd.Series(dtype=float)
    else:
        if 'Close' not in df.columns:
            return pd.Series(dtype=float)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    return pd.to_numeric(close, errors='coerce').dropna()


def clean_close(close):
    close = pd.Series(close).copy()
    close.index = pd.to_datetime(close.index, errors='coerce')
    close = close[~close.index.isna()]
    close = close[~close.index.duplicated(keep='last')]
    close = pd.to_numeric(close, errors='coerce').dropna()
    close = close[close > 0].sort_index()
    return close


def resample_close(close, tf):
    close = clean_close(close)
    if close.empty or tf == 'D':
        return close
    if tf == 'W':
        out = close.resample('W-FRI').last()
    elif tf == 'M':
        out = close.resample('ME').last()
    elif tf == '3M':
        out = close.resample('QE-DEC').last()
    elif tf == '6M':
        m = close.resample('ME').last().dropna()
        out = m[m.index.month.isin([6, 12])].copy()
        last_date = close.index[-1]
        half_end_month = 6 if last_date.month <= 6 else 12
        current_label = pd.Timestamp(year=last_date.year, month=half_end_month, day=1) + pd.offsets.MonthEnd(0)
        already = len(out) and out.index[-1].year == last_date.year and out.index[-1].month == half_end_month
        if not already:
            out.loc[current_label] = close.iloc[-1]
            out = out.sort_index()
    elif tf == '12M':
        out = close.resample('YE-DEC').last()
    else:
        raise ValueError(tf)
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
    ema_values = {}
    row = {'Hisse': symbol, 'Güncel Fiyat': price, 'Son Veri Tarihi': last_date.date()}
    for tf in TF_LIST:
        s = resample_close(close, tf)
        for length in EMA_LIST:
            name = f'{tf}_EMA{length}'
            val = ema_last(s, length)
            ema_values[name] = val
            row[name] = val
    valid = {k: v for k, v in ema_values.items() if pd.notna(v)}
    tumu_var = len(valid) == 24
    if tumu_var:
        max_name = max(valid, key=valid.get)
        max_ema = float(valid[max_name])
        tumu_ustu = all(price > v for v in valid.values())
        dist = ((price / max_ema) - 1) * 100 if max_ema > 0 and price > max_ema else np.nan
        max3 = bool(tumu_ustu and pd.notna(dist) and 0 <= dist <= MAX_UZAKLIK_PCT)
    else:
        max_name, max_ema, tumu_ustu, dist, max3 = '', np.nan, False, np.nan, False
    row.update({
        '24 EMA TÜMÜ VAR': tumu_var,
        '24 EMA TÜMÜ ÜSTÜ': tumu_ustu,
        'EN YÜKSEK EMA MAX %3': max3,
        'En Yüksek EMA Adı': max_name,
        'En Yüksek EMA': max_ema,
        'En Yüksek EMA Uzaklık %': dist,
    })
    return row


def run_scan(symbols, progress_cb=None):
    results, errors = [], []
    yahoo_symbols = [f'{s}.IS' for s in symbols]
    total = len(yahoo_symbols)
    for start in range(0, total, CHUNK_SIZE):
        chunk = yahoo_symbols[start:start+CHUNK_SIZE]
        try:
            data = yf.download(
                tickers=chunk,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=False,
                actions=False,
                group_by='column',
                threads=True,
                progress=False,
                timeout=30,
            )
        except Exception as e:
            for ys in chunk:
                errors.append({'Hisse': ys.replace('.IS',''), 'Hata': str(e)})
            continue
        for ys in chunk:
            symbol = ys.replace('.IS','')
            try:
                close = normalize_downloaded_close(data, ys)
                if close.empty:
                    errors.append({'Hisse': symbol, 'Hata': 'Close verisi yok'})
                    continue
                row = analyze_symbol(close, symbol)
                if row:
                    results.append(row)
            except Exception as e:
                errors.append({'Hisse': symbol, 'Hata': str(e)})
        done = min(start + CHUNK_SIZE, total)
        if progress_cb:
            progress_cb(done / max(total,1), f'{done}/{total} hisse işlendi')
        time.sleep(0.35)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['EN YÜKSEK EMA MAX %3','En Yüksek EMA Uzaklık %','Hisse'], ascending=[False,True,True], na_position='last').reset_index(drop=True)
    return df, pd.DataFrame(errors)

with st.sidebar:
    st.subheader('Tarama Ayarları')
    st.caption('Veri: Yahoo Finance • ADJ: KAPALI')
    max_dist = st.number_input('Maksimum EMA uzaklığı %', min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    st.caption('İlk sürümde ana hesaplama %3 mantığına göre sabittir. Bu alan sonraki sürümde dinamik yapılabilir.')

run = st.button('TARAMAYI BAŞLAT', type='primary', use_container_width=True)

if run:
    try:
        with st.spinner('BIST sembolleri alınıyor...'):
            symbols = get_bist_symbols()
        progress = st.progress(0.0, text='Tarama başlıyor...')
        df, err = run_scan(symbols, lambda p, t: progress.progress(min(p,1.0), text=t))

        passed_mask = df['EN YÜKSEK EMA MAX %3'] == True if not df.empty else pd.Series(dtype=bool)
        passed_df = df[passed_mask].copy() if not df.empty else pd.DataFrame()

        if not passed_df.empty:
            progress.progress(0.0, text='EMA geçen hisselere 5Y P/R4 hesaplanıyor...')
            passed_with_levels, level_errors = add_5y_p_r4(
                passed_df,
                lambda p, t: progress.progress(min(p,1.0), text=t)
            )

            # Hesaplanan P/R4 kolonlarını ana dataframe'e hisse bazında geri yaz.
            level_cols = [
                c for c in passed_with_levels.columns
                if c not in df.columns or c == 'Hisse'
            ]
            level_part = passed_with_levels[level_cols].copy()
            df = df.merge(level_part, on='Hisse', how='left')

            if level_errors:
                level_err_df = pd.DataFrame(level_errors)
                if err is None or err.empty:
                    err = level_err_df
                else:
                    if 'Aşama' not in err.columns:
                        err = err.copy()
                        err.insert(0, 'Aşama', 'EMA')
                    err = pd.concat([err, level_err_df], ignore_index=True)

        progress.empty()
        st.session_state['scan_df'] = df
        st.session_state['err_df'] = err
        st.session_state['scan_time'] = datetime.now(ZoneInfo('Europe/Istanbul')).strftime('%d.%m.%Y %H:%M')
    except Exception as e:
        st.error(f'Tarama başlatılamadı: {e}')

if 'scan_df' in st.session_state:
    df = st.session_state['scan_df']
    err = st.session_state.get('err_df', pd.DataFrame())
    passed = df[df['EN YÜKSEK EMA MAX %3'] == True].copy() if not df.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Taranan Hisse', len(df))
    c2.metric('24 EMA Üstü', int(df['24 EMA TÜMÜ ÜSTÜ'].sum()) if not df.empty else 0)
    c3.metric('MAX %3 Geçen', len(passed))
    c4.metric('Son Tarama', st.session_state.get('scan_time','-'))

    st.subheader('Sonuçlar')
    st.caption("Sıralama: 5Y Pivot (P)'ye en yakın hisseden en uzağa. P Uzaklık % negatifse P altında, pozitifse P üstünde.")
    q = st.text_input('Hisse ara', placeholder='Örn: THYAO')
    show = passed.copy()
    if q and not show.empty:
        show = show[show['Hisse'].str.contains(q.upper(), na=False)]

    # Hocanın istediği ana bakış:
    # EMA filtresinden geçenleri 5Y Pivot (P)'ye EN YAKINDAN uzağa sırala.
    # Ekranda görünen "5Y P Uzaklık %" işaretli kalır:
    # negatif = P altında, pozitif = P üstünde.
    if not show.empty and '5Y P Uzaklık %' in show.columns:
        show['_P_MUTLAK_UZAKLIK'] = pd.to_numeric(
            show['5Y P Uzaklık %'], errors='coerce'
        ).abs()
        show = show.sort_values(
            ['_P_MUTLAK_UZAKLIK', 'Hisse'],
            ascending=[True, True],
            na_position='last'
        ).drop(columns=['_P_MUTLAK_UZAKLIK']).reset_index(drop=True)

    front = [
        'Hisse',
        'Güncel Fiyat',
        '5Y Camarilla P',
        '5Y P Uzaklık %',
        '5Y P Konum',
        'En Yüksek EMA Adı',
        'En Yüksek EMA',
        'En Yüksek EMA Uzaklık %',
        '5Y Camarilla R4',
        '5Y R4 Uzaklık %',
        '5Y R4 Konum',
        'Son Veri Tarihi'
    ]
    cols = front + [c for c in show.columns if c.startswith(tuple(TF_LIST)) and not c.endswith('USTUNDE')]
    cols = [c for c in cols if c in show.columns]

    # Görsel düzen: fiyat 2, EMA 4, yüzde 2 ondalık gösterilir.
    display_df = show[cols].copy()
    if 'Güncel Fiyat' in display_df.columns:
        display_df['Güncel Fiyat'] = pd.to_numeric(display_df['Güncel Fiyat'], errors='coerce').round(2)
    if 'En Yüksek EMA' in display_df.columns:
        display_df['En Yüksek EMA'] = pd.to_numeric(display_df['En Yüksek EMA'], errors='coerce').round(4)
    if 'En Yüksek EMA Uzaklık %' in display_df.columns:
        display_df['En Yüksek EMA Uzaklık %'] = pd.to_numeric(display_df['En Yüksek EMA Uzaklık %'], errors='coerce').round(2)

    for c in ['5Y Camarilla P', '5Y Camarilla R4']:
        if c in display_df.columns:
            display_df[c] = pd.to_numeric(display_df[c], errors='coerce').round(4)
    for c in ['5Y P Uzaklık %', '5Y R4 Uzaklık %']:
        if c in display_df.columns:
            display_df[c] = pd.to_numeric(display_df[c], errors='coerce').round(2)

    ema_cols = [c for c in display_df.columns if '_EMA' in c and c not in ['En Yüksek EMA Adı', 'En Yüksek EMA Uzaklık %']]
    for c in ema_cols:
        display_df[c] = pd.to_numeric(display_df[c], errors='coerce').round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Güncel Fiyat': st.column_config.NumberColumn(format='%.2f'),
            'En Yüksek EMA': st.column_config.NumberColumn(format='%.4f'),
            'En Yüksek EMA Uzaklık %': st.column_config.NumberColumn(format='%.2f'),
            '5Y Camarilla P': st.column_config.NumberColumn(format='%.4f'),
            '5Y P Uzaklık %': st.column_config.NumberColumn(format='%.2f'),
            '5Y Camarilla R4': st.column_config.NumberColumn(format='%.4f'),
            '5Y R4 Uzaklık %': st.column_config.NumberColumn(format='%.2f'),
            **{c: st.column_config.NumberColumn(format='%.4f') for c in ema_cols},
        }
    )

    # Sonuçları gerçek Excel (.xlsx) dosyası olarak indir.
    excel_df = show.copy()
    if 'Güncel Fiyat' in excel_df.columns:
        excel_df['Güncel Fiyat'] = pd.to_numeric(excel_df['Güncel Fiyat'], errors='coerce').round(2)
    if 'En Yüksek EMA' in excel_df.columns:
        excel_df['En Yüksek EMA'] = pd.to_numeric(excel_df['En Yüksek EMA'], errors='coerce').round(4)
    if 'En Yüksek EMA Uzaklık %' in excel_df.columns:
        excel_df['En Yüksek EMA Uzaklık %'] = pd.to_numeric(excel_df['En Yüksek EMA Uzaklık %'], errors='coerce').round(2)

    for c in ['5Y Camarilla P', '5Y Camarilla R4', '5Y High (TV 12M)', '5Y Low (TV 12M)', '5Y Close (TV 12M)']:
        if c in excel_df.columns:
            excel_df[c] = pd.to_numeric(excel_df[c], errors='coerce').round(4)
    for c in ['5Y P Uzaklık %', '5Y R4 Uzaklık %']:
        if c in excel_df.columns:
            excel_df[c] = pd.to_numeric(excel_df[c], errors='coerce').round(2)

    excel_ema_cols = [
        c for c in excel_df.columns
        if '_EMA' in c and c not in ['En Yüksek EMA Adı', 'En Yüksek EMA Uzaklık %']
    ]
    for c in excel_ema_cols:
        excel_df[c] = pd.to_numeric(excel_df[c], errors='coerce').round(4)

    # Excel'de de Pivot/P ilk bakışta görülsün.
    excel_front = [
        'Hisse',
        'Güncel Fiyat',
        '5Y Camarilla P',
        '5Y P Uzaklık %',
        '5Y P Konum',
        'En Yüksek EMA Adı',
        'En Yüksek EMA',
        'En Yüksek EMA Uzaklık %',
        '5Y Camarilla R4',
        '5Y R4 Uzaklık %',
        '5Y R4 Konum',
        'Son Veri Tarihi'
    ]
    excel_front = [c for c in excel_front if c in excel_df.columns]
    excel_rest = [c for c in excel_df.columns if c not in excel_front]
    excel_df = excel_df[excel_front + excel_rest]

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        excel_df.to_excel(writer, sheet_name='SONUCLAR', index=False)

        if err is not None and not err.empty:
            err.to_excel(writer, sheet_name='HATALAR', index=False)
        else:
            pd.DataFrame({'Bilgi': ['Hata yok.']}).to_excel(
                writer, sheet_name='HATALAR', index=False
            )

        ws = writer.book['SONUCLAR']
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions

        headers = {cell.value: cell.column for cell in ws[1]}

        if 'Güncel Fiyat' in headers:
            col = headers['Güncel Fiyat']
            for row in range(2, ws.max_row + 1):
                ws.cell(row, col).number_format = '0.00'

        if 'En Yüksek EMA' in headers:
            col = headers['En Yüksek EMA']
            for row in range(2, ws.max_row + 1):
                ws.cell(row, col).number_format = '0.0000'

        if 'En Yüksek EMA Uzaklık %' in headers:
            col = headers['En Yüksek EMA Uzaklık %']
            for row in range(2, ws.max_row + 1):
                ws.cell(row, col).number_format = '0.00'

        for name in ['5Y Camarilla P', '5Y Camarilla R4', '5Y High (TV 12M)', '5Y Low (TV 12M)', '5Y Close (TV 12M)']:
            if name in headers:
                col = headers[name]
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, col).number_format = '0.0000'

        for name in ['5Y P Uzaklık %', '5Y R4 Uzaklık %']:
            if name in headers:
                col = headers[name]
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, col).number_format = '0.00'

        for name in excel_ema_cols:
            if name in headers:
                col = headers[name]
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, col).number_format = '0.0000'

        for col_cells in ws.columns:
            max_len = 0
            for cell in list(col_cells)[:100]:
                val = '' if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 11), 28)

    excel_buffer.seek(0)
    excel_name = f"EMA_USTU_TARAMA_{datetime.now(ZoneInfo('Europe/Istanbul')).strftime('%Y%m%d_%H%M')}.xlsx"

    st.download_button(
        'SONUÇLARI EXCEL İNDİR',
        data=excel_buffer.getvalue(),
        file_name=excel_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

    with st.expander('Hatalar'):
        if err is None or err.empty:
            st.success('Hata yok.')
        else:
            st.dataframe(err, use_container_width=True, hide_index=True)
else:
    st.info('Taramayı başlatmak için üstteki düğmeye bas.')
    st.markdown('**Koşul:** Önce EMA filtresi uygulanır: fiyat D, W, M, 3M, 6M ve 12M periyotlarındaki EMA 5/21/50/200 değerlerinin tamamının üzerinde ve en yüksek EMA’dan en fazla %3 yukarıda olmalıdır. Ardından yalnızca bu hisselerin TradingView 12M verisiyle 5Y Pivot (P) uzaklığı hesaplanır ve sonuçlar P’ye en yakından uzağa sıralanır. R4 bilgisi ek bilgi olarak korunur.')
