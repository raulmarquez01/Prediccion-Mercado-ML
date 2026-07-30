# -*- coding: utf-8 -*-
"""
============================================================================
PROYECTO FINAL — DATA SCIENCE & MACHINE LEARNING (v3)
Fase 3: Aplicación web — Predicción individual + Cartera multi-horizonte
============================================================================
Dos modos:
  1. Prediccion individual (v2): un ticker, direccion de manana (clasificacion)
  2. Cartera (v3, nuevo): varias acciones + horizonte (1sem/1mes/3mes/6mes)
     + monto manual por accion -> rendimiento % proyectado y monto nominal
     esperado al final del periodo, usando los modelos de regresion de la
     Fase B (modelo_horizonte_{5,21,63,126}d.joblib).

Ejecutar en local:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictor de Direccion de Mercado",
                   page_icon="📈", layout="centered")

NAVY = "#1E2761"

# ----------------------------------------------------------------------
# Universo de 60 tickers: sector + estilo (Growth/Value/Index)
# Extraido de mercado.db (Fase A). Necesario porque los modelos
# multi-horizonte usan 'sector' y 'estilo' como variables categoricas.
# ----------------------------------------------------------------------
TICKERS_META = {
    'AAPL': ('Technology', 'Growth'), 'ABBV': ('Healthcare', 'Growth'),
    'ADBE': ('Technology', 'Value'), 'AMT': ('Real Estate', 'Growth'),
    'AMZN': ('Consumer Cyclical', 'Growth'), 'APD': ('Basic Materials', 'Growth'),
    'AXP': ('Financial Services', 'Value'), 'BA': ('Industrials', 'Growth'),
    'BAC': ('Financial Services', 'Value'), 'CAT': ('Industrials', 'Growth'),
    'CMCSA': ('Communication Services', 'Value'), 'COP': ('Energy', 'Value'),
    'COST': ('Consumer Defensive', 'Growth'), 'CRM': ('Technology', 'Value'),
    'CSCO': ('Technology', 'Growth'), 'CVX': ('Energy', 'Value'),
    'DIS': ('Communication Services', 'Value'), 'DUK': ('Utilities', 'Value'),
    'GE': ('Industrials', 'Growth'), 'GOOGL': ('Communication Services', 'Value'),
    'GS': ('Financial Services', 'Value'), 'HD': ('Consumer Cyclical', 'Growth'),
    'HON': ('Industrials', 'Value'), 'IBM': ('Technology', 'Value'),
    'JNJ': ('Healthcare', 'Growth'), 'JPM': ('Financial Services', 'Value'),
    'KO': ('Consumer Defensive', 'Growth'), 'LIN': ('Basic Materials', 'Growth'),
    'LLY': ('Healthcare', 'Growth'), 'LOW': ('Consumer Cyclical', 'Value'),
    'MCD': ('Consumer Cyclical', 'Growth'), 'META': ('Communication Services', 'Growth'),
    'MRK': ('Healthcare', 'Growth'), 'MS': ('Financial Services', 'Value'),
    'MSFT': ('Technology', 'Growth'), 'NEE': ('Utilities', 'Value'),
    'NEM': ('Basic Materials', 'Value'), 'NFLX': ('Communication Services', 'Growth'),
    'NKE': ('Consumer Cyclical', 'Value'), 'NVDA': ('Technology', 'Growth'),
    'ORCL': ('Technology', 'Growth'), 'PEP': ('Consumer Defensive', 'Value'),
    'PFE': ('Healthcare', 'Value'), 'PG': ('Consumer Defensive', 'Growth'),
    'PLD': ('Real Estate', 'Growth'), 'QCOM': ('Technology', 'Value'),
    'SBUX': ('Consumer Cyclical', 'Growth'), 'SLB': ('Energy', 'Value'),
    'SO': ('Utilities', 'Growth'), 'SPG': ('Real Estate', 'Growth'),
    'SPY': ('Index ETF', 'Index'), 'T': ('Communication Services', 'Value'),
    'TGT': ('Consumer Cyclical', 'Value'), 'TXN': ('Technology', 'Growth'),
    'UNH': ('Healthcare', 'Growth'), 'UPS': ('Industrials', 'Value'),
    'VZ': ('Communication Services', 'Value'), 'WFC': ('Financial Services', 'Value'),
    'WMT': ('Consumer Defensive', 'Growth'), 'XOM': ('Energy', 'Value'),
}

FEATURES_NUM = ['ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10',
                'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
                'rsi_14', 'bb_ancho', 'bb_pctb', 'atr_14_norm',
                'vol_cambio_pct', 'vol_relativo', 'rango_diario']
FEATURES_CAT = ['dia_semana', 'sector', 'estilo']
DIAS = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes'}

HORIZONTES = {'1 semana (5 dias habiles)': 5, '1 mes (21 dias habiles)': 21,
             '3 meses (63 dias habiles)': 63, '6 meses (126 dias habiles)': 126}

GLOSARIO = {
    'ret_lag_1':  ('Retorno de ayer (momentum 1 dia)', lambda v: 'Subio ayer' if v > 0 else 'Bajo ayer'),
    'ret_lag_2':  ('Retorno de hace 2 dias', lambda v: 'Positivo' if v > 0 else 'Negativo'),
    'ret_lag_3':  ('Retorno de hace 3 dias', lambda v: 'Positivo' if v > 0 else 'Negativo'),
    'ret_lag_5':  ('Retorno de hace 5 dias', lambda v: 'Positivo' if v > 0 else 'Negativo'),
    'ret_lag_10': ('Retorno de hace 10 dias', lambda v: 'Positivo' if v > 0 else 'Negativo'),
    'dist_sma_5':  ('Distancia a la media de 5 dias',
                    lambda v: 'Sobre su tendencia corta' if v > 0.02 else ('Bajo su tendencia corta' if v < -0.02 else 'En linea con su tendencia')),
    'dist_sma_10': ('Distancia a la media de 10 dias',
                    lambda v: 'Estirado al alza (riesgo de reversion)' if v > 0.03 else ('Estirado a la baja' if v < -0.03 else 'Cerca de su media')),
    'dist_sma_20': ('Distancia a la media de 20 dias',
                    lambda v: 'Tendencia alcista sostenida' if v > 0.03 else ('Tendencia bajista sostenida' if v < -0.03 else 'Sin tendencia clara')),
    'dist_sma_50': ('Distancia a la media de 50 dias',
                    lambda v: 'Tendencia alcista de fondo' if v > 0.05 else ('Tendencia bajista de fondo' if v < -0.05 else 'Neutral')),
    'rsi_14':     ('RSI - Indice de Fuerza Relativa (14 dias)',
                    lambda v: 'Sobrecomprado (>70)' if v > 70 else ('Sobrevendido (<30)' if v < 30 else 'Zona neutral (30-70)')),
    'bb_ancho':   ('Ancho de Bandas de Bollinger (volatilidad)',
                    lambda v: 'Volatilidad alta' if v > 0.10 else 'Volatilidad contenida'),
    'bb_pctb':    ('Posicion dentro de las Bandas de Bollinger',
                    lambda v: 'Cerca del techo del canal' if v > 0.8 else ('Cerca del piso del canal' if v < 0.2 else 'Centro del canal')),
    'atr_14_norm':('ATR - Rango verdadero promedio (14 dias)',
                    lambda v: f'~{v*100:.1f}% de movimiento diario tipico'),
    'vol_cambio_pct': ('Cambio de volumen vs. ayer',
                    lambda v: 'Actividad inusualmente alta' if v > 0.5 else ('Actividad baja' if v < -0.3 else 'Actividad normal')),
    'vol_relativo': ('Volumen relativo a su media de 20 dias',
                    lambda v: 'Muy por encima de lo normal' if v > 1.5 else ('Por debajo de lo normal' if v < 0.7 else 'Normal')),
    'rango_diario': ('Amplitud de la sesion de hoy',
                    lambda v: 'Sesion muy volatil' if v > 0.03 else 'Sesion tranquila'),
}


@st.cache_resource
def cargar_modelo_direccion():
    return joblib.load('modelo_final.joblib')

@st.cache_resource
def cargar_modelo_horizonte(dias):
    return joblib.load(f'modelo_horizonte_{dias}d.joblib')


@st.cache_data(ttl=1800)
def descargar_datos(ticker: str) -> pd.DataFrame:
    fin = datetime.today()
    inicio = fin - timedelta(days=160)
    df = yf.download(ticker, start=inicio.strftime('%Y-%m-%d'),
                     end=fin.strftime('%Y-%m-%d'), interval='1d',
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()


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
    sector, estilo = TICKERS_META[ticker]
    g['sector'] = sector
    g['estilo'] = estilo
    return g


def obtener_features_actuales(ticker):
    datos = descargar_datos(ticker)
    if len(datos) < 60:
        return None
    feats = construir_features(datos, ticker)
    ultima = feats.dropna(subset=FEATURES_NUM).iloc[[-1]]
    return ultima if not ultima.empty else None


# ======================================================================
# INTERFAZ
# ======================================================================
st.markdown(f"<h1 style='color:{NAVY};'>📈 Predictor de Direccion de Mercado</h1>",
            unsafe_allow_html=True)
st.caption("Clasificacion y regresion multi-horizonte con indicadores tecnicos · "
           "Proyecto Final Data Science & ML — 4Geeks Academy")

modo = st.radio("¿Que quieres hacer?",
                ["🔮 Prediccion individual (manana)", "💼 Armar una cartera"],
                horizontal=True)

TICKERS = sorted(TICKERS_META.keys())

# ========================================================================
# MODO 1 — Prediccion individual (direccion a 1 dia, v2 sin cambios)
# ========================================================================
if modo == "🔮 Prediccion individual (manana)":
    ticker = st.selectbox("Selecciona un ticker", TICKERS,
                          format_func=lambda t: f"{t}  ·  {TICKERS_META[t][0]}")

    if st.button("🔮 Predecir direccion de manana", type="primary"):
        with st.spinner("Descargando datos recientes y calculando indicadores..."):
            ultima = obtener_features_actuales(ticker)
            if ultima is None:
                st.error("No se pudo obtener informacion suficiente para este ticker.")
                st.stop()
            modelo = cargar_modelo_direccion()
            X = ultima[FEATURES_NUM + ['dia_semana', 'sector']]
            proba_sube = float(modelo.predict_proba(X)[0, 1])
            pred = int(proba_sube >= 0.5)

        fecha_dato = pd.to_datetime(ultima['date'].iloc[0]).date()
        precio = float(ultima['close'].iloc[0])

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Ultimo cierre", f"${precio:,.2f}", help=f"Dato del {fecha_dato}")
        c2.metric("Prediccion", "⬆️ SUBE" if pred == 1 else "⬇️ BAJA / IGUAL")
        c3.metric("Probabilidad de subida", f"{proba_sube*100:.1f}%")
        st.progress(proba_sube, text=f"Confianza del modelo en 'Sube': {proba_sube*100:.1f}%")

        if abs(proba_sube - 0.5) < 0.05:
            st.info("La probabilidad esta cerca del 50%: el modelo no tiene una "
                    "senal fuerte para este activo hoy.")

        st.subheader("Indicadores usados por el modelo (dia mas reciente)")
        filas = []
        for cod in FEATURES_NUM:
            val = float(ultima[cod].iloc[0])
            nombre, interpretar = GLOSARIO[cod]
            filas.append({'Indicador': nombre, 'Codigo': cod,
                          'Valor': round(val, 4), 'Interpretacion': interpretar(val)})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        st.caption("📖 Ver `glosario_indicadores.md` en el repositorio para el detalle completo.")

# ========================================================================
# MODO 2 — Cartera multi-horizonte (Fase C, NUEVO)
# ========================================================================
else:
    st.subheader("Arma tu cartera hipotetica")
    st.caption("Elige varias acciones, un horizonte de inversion, y cuanto quieres "
               "asignar a cada una. El modelo estima el rendimiento % esperado a "
               "ese plazo para cada accion, usando indicadores tecnicos de hoy.")

    horizonte_label = st.selectbox("Horizonte de inversion", list(HORIZONTES.keys()))
    dias_horizonte = HORIZONTES[horizonte_label]

    tickers_elegidos = st.multiselect(
        "Elige las acciones de tu cartera",
        TICKERS, default=['AAPL', 'JPM', 'SPY'],
        format_func=lambda t: f"{t} · {TICKERS_META[t][0]} · {TICKERS_META[t][1]}")

    montos = {}
    if tickers_elegidos:
        st.markdown("**Monto a invertir en cada accion (USD):**")
        cols = st.columns(min(len(tickers_elegidos), 4))
        for i, tk in enumerate(tickers_elegidos):
            with cols[i % len(cols)]:
                montos[tk] = st.number_input(f"{tk}", min_value=0.0, value=1000.0,
                                             step=100.0, key=f"monto_{tk}")

    if st.button("💼 Calcular proyeccion de la cartera", type="primary"):
        if not tickers_elegidos:
            st.warning("Elige al menos una accion.")
            st.stop()

        total_invertido = sum(montos.values())
        if total_invertido <= 0:
            st.warning("Asigna un monto mayor a cero en al menos una accion.")
            st.stop()

        with st.spinner(f"Calculando proyeccion a {horizonte_label} para "
                        f"{len(tickers_elegidos)} acciones..."):
            modelo_h = cargar_modelo_horizonte(dias_horizonte)
            filas_cartera = []
            for tk in tickers_elegidos:
                ultima = obtener_features_actuales(tk)
                if ultima is None:
                    filas_cartera.append({'ticker': tk, 'error': True})
                    continue
                X = ultima[FEATURES_NUM + FEATURES_CAT]
                rendimiento_pred = float(modelo_h.predict(X)[0])
                precio_hoy = float(ultima['close'].iloc[0])
                monto = montos[tk]
                monto_final = monto * (1 + rendimiento_pred)
                precio_proyectado = precio_hoy * (1 + rendimiento_pred)
                filas_cartera.append({
                    'ticker': tk, 'error': False,
                    'sector': TICKERS_META[tk][0], 'estilo': TICKERS_META[tk][1],
                    'precio_hoy': precio_hoy, 'precio_proyectado': precio_proyectado,
                    'rendimiento_%': rendimiento_pred * 100,
                    'monto_invertido': monto, 'monto_proyectado': monto_final,
                })

        df_cartera = pd.DataFrame(filas_cartera)
        if 'error' in df_cartera.columns:
            errores = df_cartera[df_cartera['error'] == True]
            df_ok = df_cartera[df_cartera['error'] == False]
        else:
            errores = pd.DataFrame()
            df_ok = df_cartera

        if len(errores) > 0:
            st.warning(f"No se pudo obtener datos para: {', '.join(errores['ticker'])}")

        if len(df_ok) == 0:
            st.error("No se pudo calcular ninguna proyeccion.")
            st.stop()

        total_proyectado = df_ok['monto_proyectado'].sum()
        rendimiento_cartera = (total_proyectado / total_invertido - 1) * 100

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Monto total invertido", f"${total_invertido:,.2f}")
        c2.metric("Monto proyectado", f"${total_proyectado:,.2f}",
                 delta=f"{rendimiento_cartera:+.2f}%")
        c3.metric("Ganancia/perdida nominal", f"${total_proyectado - total_invertido:+,.2f}")

        st.subheader("Detalle por accion")
        tabla = df_ok[['ticker', 'sector', 'estilo', 'precio_hoy', 'precio_proyectado',
                       'rendimiento_%', 'monto_invertido', 'monto_proyectado']].copy()
        tabla.columns = ['Ticker', 'Sector', 'Estilo', 'Precio hoy', 'Precio proyectado',
                         'Rendimiento %', 'Invertido ($)', 'Proyectado ($)']
        st.dataframe(tabla.style.format({
            'Precio hoy': '${:,.2f}', 'Precio proyectado': '${:,.2f}',
            'Rendimiento %': '{:+.2f}%', 'Invertido ($)': '${:,.2f}',
            'Proyectado ($)': '${:,.2f}'
        }), use_container_width=True, hide_index=True)

        st.bar_chart(df_ok.set_index('ticker')['rendimiento_%'])

        st.warning(
            "⚠️ **Esta proyeccion es un ejercicio academico, no asesoria de inversion.** "
            "Los modelos de horizontes largos tienen R2 modesto y sus rendimientos se "
            "superponen entre fechas consecutivas, lo que puede optimizar artificialmente "
            "las metricas de validacion. No invierta dinero real basandose unicamente "
            "en esta herramienta.")

st.divider()
st.caption("⚠️ Modelos academicos. Datos: Yahoo Finance. Esto NO es una recomendacion de inversion.")
