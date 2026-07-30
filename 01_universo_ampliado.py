# -*- coding: utf-8 -*-
"""
============================================================================
PROYECTO FINAL — DATA SCIENCE & MACHINE LEARNING (v2)
Fase A: Universo ampliado (60 tickers) + clasificación Growth / Value
============================================================================
Este script REEMPLAZA a 01_adquisicion_datos.py. Amplía el universo de 22 a
60 tickers y agrega una tercera variable categórica: el "estilo" de la
acción (Growth o Value), que habilita el insight de "¿el modelo predice
mejor las acciones Growth o las Value?" pedido para la Fase E.

METODOLOGÍA GROWTH / VALUE (documentar en la tesis/presentación):
  Para cada ticker se obtienen dos ratios fundamentales públicos vía
  yfinance: P/E (precio/utilidad) y P/B (precio/valor en libros).
  Se calcula la MEDIANA de cada ratio sobre las 60 acciones del universo.
  Una acción se clasifica como GROWTH si sus dos ratios están POR ENCIMA
  de la mediana del grupo (el mercado paga una prima por crecimiento
  esperado); se clasifica como VALUE si ambos están POR DEBAJO (se cotiza
  barata respecto a sus fundamentos actuales). Si los dos ratios discrepan
  (uno arriba y otro abajo), se usa el promedio de los dos rangos
  percentiles para desempatar. Este es el mismo principio que usan los
  índices S&P Growth/Value, simplificado para ser reproducible con datos
  gratuitos y sin suscripción.

Autor: Raúl Márquez
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print('✅ Librerías cargadas')

# ============================================================
#  CELDA 1 — Universo de 60 tickers, 11 sectores GICS
#
#  Se amplió de 22 a 60 para tener suficientes acciones por sector
#  y por horizonte de inversión al construir carteras (Fase C), y
#  suficiente muestra para comparar Growth vs Value de forma
#  estadísticamente razonable (Fase E).
# ============================================================

TICKERS_META = [
    # symbol   sector                     exchange
    ('AAPL', 'Technology', 'NASDAQ'), ('MSFT', 'Technology', 'NASDAQ'),
    ('NVDA', 'Technology', 'NASDAQ'), ('IBM', 'Technology', 'NYSE'),
    ('ORCL', 'Technology', 'NYSE'), ('CSCO', 'Technology', 'NASDAQ'),
    ('ADBE', 'Technology', 'NASDAQ'), ('CRM', 'Technology', 'NYSE'),
    ('JPM', 'Financial Services', 'NYSE'), ('GS', 'Financial Services', 'NYSE'),
    ('BAC', 'Financial Services', 'NYSE'), ('WFC', 'Financial Services', 'NYSE'),
    ('MS', 'Financial Services', 'NYSE'), ('AXP', 'Financial Services', 'NYSE'),
    ('XOM', 'Energy', 'NYSE'), ('CVX', 'Energy', 'NYSE'),
    ('COP', 'Energy', 'NYSE'), ('SLB', 'Energy', 'NYSE'),
    ('JNJ', 'Healthcare', 'NYSE'), ('PFE', 'Healthcare', 'NYSE'),
    ('UNH', 'Healthcare', 'NYSE'), ('MRK', 'Healthcare', 'NYSE'),
    ('ABBV', 'Healthcare', 'NYSE'), ('LLY', 'Healthcare', 'NYSE'),
    ('KO', 'Consumer Defensive', 'NYSE'), ('WMT', 'Consumer Defensive', 'NYSE'),
    ('PG', 'Consumer Defensive', 'NYSE'), ('PEP', 'Consumer Defensive', 'NASDAQ'),
    ('COST', 'Consumer Defensive', 'NASDAQ'),
    ('DIS', 'Communication Services', 'NYSE'), ('T', 'Communication Services', 'NYSE'),
    ('VZ', 'Communication Services', 'NYSE'), ('NFLX', 'Communication Services', 'NASDAQ'),
    ('CMCSA', 'Communication Services', 'NASDAQ'),
    ('BA', 'Industrials', 'NYSE'), ('CAT', 'Industrials', 'NYSE'),
    ('GE', 'Industrials', 'NYSE'), ('UPS', 'Industrials', 'NYSE'),
    ('HON', 'Industrials', 'NASDAQ'),
    ('HD', 'Consumer Cyclical', 'NYSE'), ('MCD', 'Consumer Cyclical', 'NYSE'),
    ('NKE', 'Consumer Cyclical', 'NYSE'), ('SBUX', 'Consumer Cyclical', 'NASDAQ'),
    ('TGT', 'Consumer Cyclical', 'NYSE'), ('LOW', 'Consumer Cyclical', 'NYSE'),
    ('LIN', 'Basic Materials', 'NYSE'), ('APD', 'Basic Materials', 'NYSE'),
    ('NEM', 'Basic Materials', 'NYSE'),
    ('NEE', 'Utilities', 'NYSE'), ('DUK', 'Utilities', 'NYSE'),
    ('SO', 'Utilities', 'NYSE'),
    ('PLD', 'Real Estate', 'NYSE'), ('AMT', 'Real Estate', 'NYSE'),
    ('SPG', 'Real Estate', 'NYSE'),
    ('AMZN', 'Consumer Cyclical', 'NASDAQ'), ('GOOGL', 'Communication Services', 'NASDAQ'),
    ('META', 'Communication Services', 'NASDAQ'), ('TXN', 'Technology', 'NASDAQ'),
    ('QCOM', 'Technology', 'NASDAQ'),
    ('SPY', 'Index ETF', 'NYSE Arca'),
]

TICKERS = [t[0] for t in TICKERS_META]
assert len(TICKERS) == len(set(TICKERS)), "Hay tickers duplicados en la lista"
print(f'Universo definido: {len(TICKERS)} tickers en '
      f'{len(set(t[1] for t in TICKERS_META))} sectores')

# ============================================================
#  CELDA 2 — Clasificación Growth / Value (ratios fundamentales)
#
#  yf.Ticker(x).info SÍ se usa aquí (a diferencia de la Fase 1),
#  porque esta clasificación se calcula UNA SOLA VEZ y se guarda en
#  la base de datos — no se repite en cada corrida del modelo ni de
#  la app, así que el costo de la llamada web es aceptable.
#  Se agrega una pausa entre llamadas para evitar rate-limiting.
# ============================================================

fundamentales = []
for tk in TICKERS:
    try:
        info = yf.Ticker(tk).info
        pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        fundamentales.append({'symbol': tk, 'pe': pe, 'pb': pb})
    except Exception as e:
        print(f'  ⚠️ {tk}: no se pudo obtener info fundamental ({e})')
        fundamentales.append({'symbol': tk, 'pe': np.nan, 'pb': np.nan})
    time.sleep(0.3)   # cortesía para no saturar la API

df_fund = pd.DataFrame(fundamentales)

# --- Corrección 1: un P/B negativo (patrimonio contable negativo, típico
# de recompras masivas de acciones o deuda alta) no es comparable en la
# misma escala que un P/B positivo — no significa "barata", significa que
# el ratio no tiene una lectura de valor estándar. Se trata como faltante.
n_pb_negativos = (df_fund['pb'] < 0).sum()
if n_pb_negativos > 0:
    print(f'⚠️  {n_pb_negativos} tickers con P/B negativo (patrimonio contable '
          f'negativo): se excluyen del ranking de P/B, se clasifican solo con P/E.')
    df_fund.loc[df_fund['pb'] < 0, 'pb'] = np.nan

n_nulos = df_fund[['pe', 'pb']].isna().any(axis=1).sum()
print(f'Fundamentales obtenidos: {len(df_fund) - n_nulos}/{len(df_fund)} '
      f'completos ({n_nulos} con algún dato faltante o P/B negativo)')

# Percentil de cada ratio dentro del universo (0 a 1)
df_fund['pct_pe'] = df_fund['pe'].rank(pct=True)
df_fund['pct_pb'] = df_fund['pb'].rank(pct=True)
df_fund['pct_promedio'] = df_fund[['pct_pe', 'pct_pb']].mean(axis=1)

# Regla documentada: percentil promedio > 0.5 => Growth, si no => Value
# Si falta algún ratio, se usa el que esté disponible; si faltan los dos,
# se asigna 'Value' por defecto (postura conservadora, no optimista)
df_fund['estilo'] = np.where(df_fund['pct_promedio'].isna(), 'Value',
                     np.where(df_fund['pct_promedio'] > 0.5, 'Growth', 'Value'))

# --- Corrección 2: un ETF de índice (SPY) no es ni Growth ni Value —
# es un promedio diversificado de cientos de empresas. Forzarlo a una
# categoría contaminaría el insight de "¿el modelo predice mejor Growth
# o Value?" con un resultado que en realidad refleja diversificación,
# no estilo de inversión. Se etiqueta aparte y se excluye de ese análisis.
df_fund.loc[df_fund['symbol'] == 'SPY', 'estilo'] = 'Index'

print('\nDistribución Growth/Value:')
print(df_fund['estilo'].value_counts().to_string())
print('\nMuestra (5 más Growth, 5 más Value):')
cols = ['symbol', 'pe', 'pb', 'estilo']
print(df_fund.sort_values('pct_promedio', ascending=False)[cols].head(5).to_string(index=False))
print(df_fund.sort_values('pct_promedio')[cols].head(5).to_string(index=False))

# ============================================================
#  CELDA 3 — Descarga de 15 años de OHLCV para los 60 tickers
#  (misma lógica que la Fase 1 original, sin cambios)
# ============================================================

FECHA_FIN = datetime.today().strftime('%Y-%m-%d')
FECHA_INICIO = (datetime.today() - timedelta(days=365 * 15)).strftime('%Y-%m-%d')

print(f'\nDescargando OHLCV diario: {FECHA_INICIO} → {FECHA_FIN} ...')
raw = yf.download(tickers=TICKERS, start=FECHA_INICIO, end=FECHA_FIN,
                  interval='1d', group_by='ticker', auto_adjust=False,
                  threads=True, progress=True)

frames, fallidos = [], []
for tk in TICKERS:
    try:
        df_tk = raw[tk].copy()
        if df_tk['Close'].dropna().empty:
            fallidos.append(tk); continue
        df_tk = df_tk.reset_index()
        df_tk['ticker'] = tk
        df_tk = df_tk.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                      'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
                             )[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        frames.append(df_tk)
    except (KeyError, TypeError):
        fallidos.append(tk)

df_prices = pd.concat(frames, ignore_index=True)
df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')
antes = len(df_prices)
df_prices = df_prices.dropna(subset=['close'])
print(f'\n✅ Descarga completa: {len(df_prices):,} filas '
      f'({antes - len(df_prices)} con close nulo eliminadas)')
if fallidos:
    print(f'⚠️ Tickers fallidos: {fallidos}')

# ============================================================
#  CELDA 4 — Base de datos: tabla tickers ahora incluye 'estilo'
# ============================================================

meta_final = pd.DataFrame(TICKERS_META, columns=['symbol', 'sector', 'exchange'])
meta_final = meta_final.merge(df_fund[['symbol', 'pe', 'pb', 'estilo']], on='symbol', how='left')

conn = sqlite3.connect('mercado.db')
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS prices')
cur.execute('DROP TABLE IF EXISTS tickers')
cur.execute('''CREATE TABLE tickers (
    symbol TEXT PRIMARY KEY, sector TEXT NOT NULL, exchange TEXT NOT NULL,
    pe REAL, pb REAL, estilo TEXT NOT NULL)''')
cur.execute('''CREATE TABLE prices (
    date TEXT NOT NULL, ticker TEXT NOT NULL, open REAL, high REAL, low REAL,
    close REAL NOT NULL, volume INTEGER,
    FOREIGN KEY (ticker) REFERENCES tickers(symbol))''')
cur.execute('CREATE INDEX idx_prices_ticker_date ON prices(ticker, date)')

meta_final.to_sql('tickers', conn, if_exists='append', index=False)
df_prices.to_sql('prices', conn, if_exists='append', index=False)
conn.commit()

n_filas = pd.read_sql('SELECT COUNT(*) AS n FROM prices', conn)['n'][0]
print(f"\n✅ Base de datos actualizada: {n_filas:,} filas de precios, "
      f"{len(meta_final)} tickers")
print(f"   Requisito ≥60,000 filas: {'✅ CUMPLE' if n_filas >= 60_000 else '❌'}")
conn.close()
print("\n✅ Fase A completa. mercado.db ahora tiene sector + estilo (Growth/Value).")
