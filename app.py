# -*- coding: utf-8 -*-
"""
============================================================================
PROYECTO FINAL — DATA SCIENCE & MACHINE LEARNING
Fase 3: Aplicación web de despliegue (Streamlit)
============================================================================
El usuario elige un ticker; la app descarga los precios más recientes vía
yfinance, reconstruye las mismas 16 variables técnicas + 2 categóricas del
entrenamiento, y consulta el modelo (Regresión Logística optimizada) para
predecir si el cierre de mañana será mayor al de hoy.

Ejecutar en local:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Configuración de página
# ----------------------------------------------------------------------
st.set_page_config(page_title="Predictor de Dirección de Mercado",
                   page_icon="📈", layout="centered")

NAVY = "#1E2761"

# ----------------------------------------------------------------------
# Universo y metadata (idéntico a la Fase 1)
# ----------------------------------------------------------------------
SECTORES = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'IBM': 'Technology', 'JPM': 'Financial Services', 'GS': 'Financial Services',
    'XOM': 'Energy', 'CVX': 'Energy', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'KO': 'Consumer Defensive', 'WMT': 'Consumer Defensive', 'PG': 'Consumer Defensive',
    'DIS': 'Communication Services', 'T': 'Communication Services',
    'VZ': 'Communication Services', 'BA': 'Industrials', 'CAT': 'Industrials',
    'HD': 'Consumer Cyclical', 'MCD': 'Consumer Cyclical', 'NKE': 'Consumer Cyclical',
    'SPY': 'Index ETF',
}

FEATURES_NUM = ['ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10',
                'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
                'rsi_14', 'bb_ancho', 'bb_pctb', 'atr_14_norm',
                'vol_cambio_pct', 'vol_relativo', 'rango_diario']
FEATURES_CAT = ['dia_semana', 'sector']

DIAS = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes'}


# ----------------------------------------------------------------------
# Carga del modelo (una sola vez, cacheado)
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_final.joblib')


# ----------------------------------------------------------------------
# Descarga de datos recientes (cacheado 30 min para no abusar de la API)
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800)
def descargar_datos(ticker: str) -> pd.DataFrame:
    fin = datetime.today()
    inicio = fin - timedelta(days=160)   # margen para 50 días hábiles de warm-up
    df = yf.download(ticker, start=inicio.strftime('%Y-%m-%d'),
                     end=fin.strftime('%Y-%m-%d'), interval='1d',
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):       # yfinance >= 0.2.40
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()


# ----------------------------------------------------------------------
# Ingeniería de variables — EXACTAMENTE la misma lógica del entrenamiento
# ----------------------------------------------------------------------
def construir_features(g: pd.DataFrame, ticker: str) -> pd.DataFrame:
    g = g.sort_values('date').copy()
    g['retorno_1d'] = g['close'].pct_change()
    for k in [1, 2, 3, 5, 10]:
        g[f'ret_lag_{k}'] = g['retorno_1d'].shift(k - 1)
    for w in [5, 10, 20, 50]:
        sma = g['close'].rolling(w).mean()
        g[f'dist_sma_{w}'] = (g['close'] - sma) / sma
    delta = g['close'].diff()
    gan = delta.clip(lower=0).rolling(14).mean()
    per = (-delta.clip(upper=0)).rolling(14).mean()
    g['rsi_14'] = 100 - (100 / (1 + gan / per.replace(0, np.nan)))
    s20 = g['close'].rolling(20).mean()
    sd20 = g['close'].rolling(20).std()
    up, lo = s20 + 2 * sd20, s20 - 2 * sd20
    g['bb_ancho'] = (up - lo) / s20
    g['bb_pctb'] = (g['close'] - lo) / (up - lo)
    tr = pd.concat([g['high'] - g['low'],
                    (g['high'] - g['close'].shift(1)).abs(),
                    (g['low'] - g['close'].shift(1)).abs()], axis=1).max(axis=1)
    g['atr_14_norm'] = tr.rolling(14).mean() / g['close']
    g['vol_cambio_pct'] = g['volume'].pct_change()
    g['vol_relativo'] = g['volume'] / g['volume'].rolling(20).mean()
    g['rango_diario'] = (g['high'] - g['low']) / g['close']
    g['dia_semana'] = pd.to_datetime(g['date']).dt.dayofweek.map(DIAS)
    g['sector'] = SECTORES[ticker]
    return g


# ======================================================================
# INTERFAZ
# ======================================================================
st.markdown(f"<h1 style='color:{NAVY};'>📈 Predictor de Dirección de Mercado</h1>",
            unsafe_allow_html=True)
st.caption("Clasificación binaria con Regresión Logística e indicadores técnicos · "
           "Proyecto Final Data Science & ML — 4Geeks Academy")

ticker = st.selectbox("Selecciona un ticker",
                      sorted(SECTORES.keys()),
                      format_func=lambda t: f"{t}  ·  {SECTORES[t]}")

if st.button("🔮 Predecir dirección de mañana", type="primary"):
    with st.spinner("Descargando datos recientes y calculando indicadores..."):
        try:
            datos = descargar_datos(ticker)
        except Exception as e:
            st.error(f"No se pudieron descargar datos de Yahoo Finance: {e}")
            st.stop()

        if len(datos) < 60:
            st.error("Historia insuficiente para calcular los indicadores (se "
                     "requieren ~50 días hábiles).")
            st.stop()

        feats = construir_features(datos, ticker)
        ultima = feats.dropna(subset=FEATURES_NUM).iloc[[-1]]   # el día más reciente

        if ultima.empty:
            st.error("No fue posible calcular los indicadores del día más reciente.")
            st.stop()

        modelo = cargar_modelo()
        X = ultima[FEATURES_NUM + FEATURES_CAT]
        proba_sube = float(modelo.predict_proba(X)[0, 1])
        pred = int(proba_sube >= 0.5)

    # ------------------ Resultado ------------------
    fecha_dato = pd.to_datetime(ultima['date'].iloc[0]).date()
    precio = float(ultima['close'].iloc[0])

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Último cierre", f"${precio:,.2f}", help=f"Dato del {fecha_dato}")
    c2.metric("Predicción", "⬆️ SUBE" if pred == 1 else "⬇️ BAJA / IGUAL")
    c3.metric("Probabilidad de subida", f"{proba_sube*100:.1f}%")

    st.progress(proba_sube,
                text=f"Confianza del modelo en 'Sube': {proba_sube*100:.1f}%")

    if abs(proba_sube - 0.5) < 0.05:
        st.info("La probabilidad está cerca del 50%: el modelo no tiene una "
                "señal fuerte para este activo hoy. En el análisis del proyecto, "
                "las señales de baja confianza aciertan menos.")

    # ------------------ Contexto ------------------
    st.subheader("Precio de cierre — últimos 120 días")
    st.line_chart(feats.set_index('date')['close'])

    st.subheader("Indicadores usados por el modelo (día más reciente)")
    tabla = ultima[FEATURES_NUM].T.rename(columns={ultima.index[0]: 'valor'})
    tabla['valor'] = tabla['valor'].astype(float).round(4)
    st.dataframe(tabla, use_container_width=True)

    with st.expander("¿Cómo decide el modelo? — variables más influyentes"):
        prep = modelo.named_steps['prep']
        clf = modelo.named_steps['clf']
        nombres = FEATURES_NUM + list(
            prep.named_transformers_['cat'].get_feature_names_out(FEATURES_CAT))
        coefs = (pd.DataFrame({'variable': nombres, 'coeficiente': clf.coef_[0]})
                 .assign(mag=lambda d: d['coeficiente'].abs())
                 .sort_values('mag', ascending=False).head(10)
                 .drop(columns='mag').set_index('variable'))
        st.bar_chart(coefs)
        st.caption("Coeficiente positivo: empuja la predicción hacia 'Sube'. "
                   "Los mayores: pertenecer al Index ETF (+), la distancia a la "
                   "SMA de 10 días (−, reversión a la media) y el día jueves (+).")

st.divider()
st.caption("⚠️ Modelo académico con desempeño cercano al azar (ROC-AUC 0.496 en "
           "16,346 días fuera de muestra). Esto NO es una recomendación de "
           "inversión. Datos: Yahoo Finance.")
