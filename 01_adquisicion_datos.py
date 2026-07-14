# -*- coding: utf-8 -*-
"""
============================================================================
PROYECTO FINAL — DATA SCIENCE & MACHINE LEARNING
Fase 1: Adquisición de datos y almacenamiento en base de datos
============================================================================

OBJETIVO DE ESTA FASE (Pasos 2 y 3 de la rúbrica):
  - Adquirir datos mediante la explotación de una API pública (yfinance,
    que consume la API de Yahoo Finance). Esto cumple el requisito de que
    los datos NO provengan de un CSV plano preexistente.
  - Almacenar los datos en una base de datos (SQLite) y demostrar el uso
    de SQL: SELECT, JOIN e INSERT.

POR QUÉ SQLITE:
  - Es una base de datos relacional completa, soporta SQL estándar,
    y no requiere levantar un servidor (ideal para Codespaces/Colab).
  - A diferencia de un CSV, permite consultas estructuradas, integridad
    de tipos y relaciones entre tablas (prices ↔ tickers).

Autor: Raúl [tu apellido]
"""

# ============================================================
#  CELDA 1 — Instalación e importaciones
# ============================================================
# yfinance: wrapper de la API pública de Yahoo Finance.
# sqlite3: viene incluido en la librería estándar de Python.
!pip install yfinance -q

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print('✅ Librerías cargadas: yfinance, pandas, sqlite3')

# ============================================================
#  CELDA 2 — Universo de 22 tickers, diversificado por sector
#
#  Criterios de selección:
#    1. Alta liquidez (todas son large caps de NYSE/NASDAQ)
#    2. Historia completa de 15+ años (evita huecos en el panel)
#    3. Diversificación sectorial → la variable 'sector' será
#       una de nuestras variables CATEGÓRICAS predictoras
# ============================================================

# Hardcodeamos sector y exchange en lugar de usar yf.Ticker(x).info
# porque .info hace una llamada web por ticker, es lenta y falla con
# frecuencia por rate-limiting. La metadata sectorial es estable en
# el tiempo, así que definirla manualmente es más robusto y reproducible.
TICKERS_META = [
    # symbol   sector                    exchange
    ('AAPL',  'Technology',             'NASDAQ'),
    ('MSFT',  'Technology',             'NASDAQ'),
    ('NVDA',  'Technology',             'NASDAQ'),
    ('IBM',   'Technology',             'NYSE'),
    ('JPM',   'Financial Services',     'NYSE'),
    ('GS',    'Financial Services',     'NYSE'),
    ('XOM',   'Energy',                 'NYSE'),
    ('CVX',   'Energy',                 'NYSE'),
    ('JNJ',   'Healthcare',             'NYSE'),
    ('PFE',   'Healthcare',             'NYSE'),
    ('KO',    'Consumer Defensive',     'NYSE'),
    ('WMT',   'Consumer Defensive',     'NYSE'),
    ('PG',    'Consumer Defensive',     'NYSE'),
    ('DIS',   'Communication Services', 'NYSE'),
    ('T',     'Communication Services', 'NYSE'),
    ('VZ',    'Communication Services', 'NYSE'),
    ('BA',    'Industrials',            'NYSE'),
    ('CAT',   'Industrials',            'NYSE'),
    ('HD',    'Consumer Cyclical',      'NYSE'),
    ('MCD',   'Consumer Cyclical',      'NYSE'),
    ('NKE',   'Consumer Cyclical',      'NYSE'),
    ('SPY',   'Index ETF',              'NYSE Arca'),
]

TICKERS = [t[0] for t in TICKERS_META]
print(f'Universo definido: {len(TICKERS)} tickers en '
      f'{len(set(t[1] for t in TICKERS_META))} sectores')

# ============================================================
#  CELDA 3 — Descarga de 15 años de datos diarios (OHLCV)
#
#  yf.download() con lista de tickers hace la descarga en batch,
#  que es mucho más eficiente que un loop de llamadas individuales.
#  auto_adjust=False para conservar Close y Adj Close por separado
#  (usaremos Close simple para mantener coherencia con volumen).
# ============================================================

FECHA_FIN    = datetime.today().strftime('%Y-%m-%d')
FECHA_INICIO = (datetime.today() - timedelta(days=365 * 15)).strftime('%Y-%m-%d')

print(f'Descargando OHLCV diario: {FECHA_INICIO} → {FECHA_FIN} ...')

raw = yf.download(
    tickers=TICKERS,
    start=FECHA_INICIO,
    end=FECHA_FIN,
    interval='1d',
    group_by='ticker',
    auto_adjust=False,
    threads=True,
    progress=True,
)

# --- Reestructurar de formato "wide" (multi-índice por ticker) a
#     formato "long" (una fila por fecha-ticker), que es el formato
#     natural para una tabla relacional en la base de datos. ---
frames = []
fallidos = []

for tk in TICKERS:
    try:
        df_tk = raw[tk].copy()
        # Si el ticker no descargó nada, todas sus columnas son NaN
        if df_tk['Close'].dropna().empty:
            fallidos.append(tk)
            continue
        df_tk = df_tk.reset_index()
        df_tk['ticker'] = tk
        df_tk = df_tk.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        frames.append(df_tk)
    except (KeyError, TypeError):
        fallidos.append(tk)

df_prices = pd.concat(frames, ignore_index=True)
df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')

# Eliminamos filas donde el precio de cierre es nulo (días sin cotización
# válida). Es el único filtro de limpieza en esta fase: la limpieza
# profunda corresponde al EDA (Fase 2).
antes = len(df_prices)
df_prices = df_prices.dropna(subset=['close'])
print(f'\n✅ Descarga completa: {len(df_prices):,} filas '
      f'({antes - len(df_prices)} filas con close nulo eliminadas)')

if fallidos:
    print(f'⚠️  Tickers que fallaron y quedaron fuera: {fallidos}')
else:
    print('✅ Los 22 tickers descargaron correctamente')

# ============================================================
#  CELDA 4 — Creación de la base de datos SQLite (mercado.db)
#
#  Modelo de datos:
#    tickers (symbol PK, sector, exchange)   ← tabla de dimensión
#    prices  (date, ticker FK, open, high,
#             low, close, volume)            ← tabla de hechos
#
#  Esta separación dimensión/hechos es la práctica estándar en
#  almacenamiento analítico: la metadata no se repite en cada fila
#  de precios, y el JOIN nos permite enriquecer cuando haga falta.
# ============================================================

conn = sqlite3.connect('mercado.db')
cur = conn.cursor()

# DROP para que el script sea re-ejecutable de cero (idempotente)
cur.execute('DROP TABLE IF EXISTS prices')
cur.execute('DROP TABLE IF EXISTS tickers')
cur.execute('DROP TABLE IF EXISTS retornos_diarios')

cur.execute('''
    CREATE TABLE tickers (
        symbol   TEXT PRIMARY KEY,
        sector   TEXT NOT NULL,
        exchange TEXT NOT NULL
    )
''')

cur.execute('''
    CREATE TABLE prices (
        date   TEXT NOT NULL,
        ticker TEXT NOT NULL,
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL NOT NULL,
        volume INTEGER,
        FOREIGN KEY (ticker) REFERENCES tickers(symbol)
    )
''')

# Índice compuesto: casi todas nuestras queries filtran por ticker+fecha
cur.execute('CREATE INDEX idx_prices_ticker_date ON prices(ticker, date)')

# --- Poblar tabla de metadata con INSERT parametrizado ---
cur.executemany(
    'INSERT INTO tickers (symbol, sector, exchange) VALUES (?, ?, ?)',
    TICKERS_META
)

# --- Poblar tabla de precios (to_sql hace el bulk insert) ---
df_prices.to_sql('prices', conn, if_exists='append', index=False)
conn.commit()

print(f"✅ Base de datos 'mercado.db' creada")
print(f"   - tickers: {cur.execute('SELECT COUNT(*) FROM tickers').fetchone()[0]} filas")
print(f"   - prices : {cur.execute('SELECT COUNT(*) FROM prices').fetchone()[0]:,} filas")

# ============================================================
#  CELDA 5 — Query 1: SELECT con filtro de fechas
#
#  Valor analítico: verificar los precios de cierre más recientes
#  de un ticker específico antes de construir features sobre ellos.
# ============================================================

q1 = '''
    SELECT date, ticker, close, volume
    FROM prices
    WHERE ticker = 'AAPL'
      AND date >= date('now', '-30 days')
    ORDER BY date DESC
'''
df_q1 = pd.read_sql(q1, conn)
print('=' * 70)
print('QUERY 1 — SELECT: últimos 30 días de AAPL')
print('=' * 70)
print(df_q1.to_string(index=False))

# ============================================================
#  CELDA 6 — Query 2: JOIN entre prices y tickers, agrupado por sector
#
#  Valor analítico: primera vista del comportamiento por sector.
#  Este es el JOIN que justifica el diseño dimensión/hechos, y la
#  variable 'sector' será una predictora categórica en el modelo.
# ============================================================

q2 = '''
    SELECT
        t.sector,
        COUNT(DISTINCT p.ticker)          AS n_tickers,
        COUNT(*)                          AS n_observaciones,
        ROUND(AVG(p.close), 2)            AS precio_promedio,
        ROUND(AVG(p.volume) / 1e6, 2)     AS volumen_promedio_M,
        MIN(p.date)                       AS primera_fecha,
        MAX(p.date)                       AS ultima_fecha
    FROM prices p
    JOIN tickers t ON p.ticker = t.symbol
    GROUP BY t.sector
    ORDER BY n_observaciones DESC
'''
df_q2 = pd.read_sql(q2, conn)
print('=' * 70)
print('QUERY 2 — JOIN prices ↔ tickers, resumen por sector')
print('=' * 70)
print(df_q2.to_string(index=False))

# ============================================================
#  CELDA 7 — Query 3: INSERT de tabla derivada (retornos diarios)
#
#  Calculamos el retorno diario de cada ticker con la función de
#  ventana LAG() de SQL y lo materializamos en una tabla nueva.
#  Valor analítico: los retornos (no los precios) son la base de
#  todo el análisis estadístico de la Fase 2, porque los precios
#  no son estacionarios pero los retornos sí (lo verificaremos
#  formalmente con el test ADF en el EDA).
# ============================================================

cur.execute('''
    CREATE TABLE retornos_diarios AS
    SELECT
        date,
        ticker,
        close,
        ROUND(
            (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
            / LAG(close) OVER (PARTITION BY ticker ORDER BY date) * 100,
        4) AS retorno_pct
    FROM prices
''')
conn.commit()

q3 = '''
    SELECT ticker,
           COUNT(retorno_pct)              AS n_retornos,
           ROUND(AVG(retorno_pct), 4)      AS retorno_medio_pct,
           ROUND(MIN(retorno_pct), 2)      AS peor_dia_pct,
           ROUND(MAX(retorno_pct), 2)      AS mejor_dia_pct
    FROM retornos_diarios
    GROUP BY ticker
    ORDER BY retorno_medio_pct DESC
'''
df_q3 = pd.read_sql(q3, conn)
print('=' * 70)
print('QUERY 3 — INSERT (CREATE TABLE AS SELECT): tabla retornos_diarios')
print('=' * 70)
print(df_q3.to_string(index=False))

# ============================================================
#  CELDA 8 — Verificación final de integridad
#
#  Checklist antes de dar por cerrada la fase de adquisición:
#    a) Volumen total de filas (objetivo: ≥ 60,000 según rúbrica)
#    b) Rango de fechas cubierto
#    c) Nulos en columnas críticas de precio
# ============================================================

n_filas   = pd.read_sql('SELECT COUNT(*) AS n FROM prices', conn)['n'][0]
rango     = pd.read_sql('SELECT MIN(date) AS ini, MAX(date) AS fin FROM prices', conn)
nulos     = pd.read_sql('''
    SELECT
        SUM(CASE WHEN open   IS NULL THEN 1 ELSE 0 END) AS nulos_open,
        SUM(CASE WHEN high   IS NULL THEN 1 ELSE 0 END) AS nulos_high,
        SUM(CASE WHEN low    IS NULL THEN 1 ELSE 0 END) AS nulos_low,
        SUM(CASE WHEN close  IS NULL THEN 1 ELSE 0 END) AS nulos_close,
        SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS nulos_volume
    FROM prices
''', conn)

print('=' * 70)
print('VERIFICACIÓN FINAL — Fase de adquisición')
print('=' * 70)
print(f"  📊 Total de filas en prices : {n_filas:,}")
print(f"  📅 Rango de fechas          : {rango['ini'][0]} → {rango['fin'][0]}")
print(f"  🔍 Nulos por columna:")
print(nulos.to_string(index=False))

cumple = '✅ CUMPLE' if n_filas >= 60_000 else '❌ NO CUMPLE — ampliar tickers o años'
print(f"\n  Requisito de la rúbrica (≥ 60,000 filas): {cumple}")

conn.close()
print("\n✅ Conexión cerrada. Base de datos lista para la Fase 2 (EDA y features).")
