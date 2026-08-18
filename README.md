# EMA ÜSTÜ TARAMA

Bağımsız Streamlit uygulaması.

## Çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Mantık
- BIST hisseleri TradingView Türkiye scanner listesinden alınır.
- Fiyat verisi Yahoo Finance üzerinden çekilir.
- `auto_adjust=False` yani ADJ KAPALI.
- D / W / M / 3M / 6M / 12M
- EMA 5 / 21 / 50 / 200
- Toplam 24 EMA
- Güncel fiyat 24 EMA'nın tamamının üzerinde olmalı.
- Fiyat en yüksek EMA'dan en fazla %3 yukarıda olmalı.

Bu proje diğer uygulamalardan bağımsızdır.
