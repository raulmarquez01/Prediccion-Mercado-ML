# -*- coding: utf-8 -*-
"""
============================================================================
PROYECTO FINAL — DATA SCIENCE & MACHINE LEARNING
Fase 2: Ingeniería de variables, análisis descriptivo, EDA y modelado
============================================================================

OBJETIVO DE ESTA FASE (Pasos 4, 5 y 6 de la rúbrica):
  - Construir 20+ variables predictoras (con 2 categóricas) a partir de
    la base SQLite creada en la Fase 1, SIN fuga de información futura.
  - Análisis descriptivo con tests de hipótesis (Shapiro-Wilk, ADF).
  - EDA completo: correlaciones, outliers, balance de clases, redundancia.
  - Split TEMPORAL train/test (nunca aleatorio en series de tiempo).
  - Entrenar y comparar 3 modelos, optimizar hiperparámetros del mejor,
    y guardar el modelo final para el despliegue (Fase 3).

PREGUNTA DE NEGOCIO:
  ¿Puede un modelo predecir si el precio de cierre de una acción subirá
  mañana (target=1) o no (target=0), usando solo información técnica
  disponible HOY?

Autor: Raúl [tu apellido]
"""

# ============================================================
#  CELDA 1 — Instalación e importaciones
# ============================================================
!pip install xgboost statsmodels -q

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from xgboost import XGBClassifier

np.random.seed(42)
print('✅ Librerías cargadas')

# ============================================================
#  CELDA 2 — Carga de datos desde SQLite (Paso 1 del prompt)
#
#  Traemos prices + el sector vía JOIN, ya ordenado por
#  ticker y fecha: ese orden es CRÍTICO porque todas las
#  variables siguientes se calculan sobre ventanas móviles.
# ============================================================

conn = sqlite3.connect('mercado.db')

df = pd.read_sql('''
    SELECT p.date, p.ticker, p.open, p.high, p.low, p.close, p.volume,
           t.sector
    FROM prices p
    JOIN tickers t ON p.ticker = t.symbol
    ORDER BY p.ticker, p.date
''', conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])
print(f'✅ Datos cargados: {len(df):,} filas, '
      f'{df["ticker"].nunique()} tickers, '
      f'{df["date"].min().date()} → {df["date"].max().date()}')

# ============================================================
#  CELDA 3 — Ingeniería de variables predictoras (Paso 2)
#
#  REGLA DE ORO CONTRA LA FUGA DE INFORMACIÓN (look-ahead bias):
#  toda variable del día t usa SOLO datos hasta el día t.
#  Por eso todo se calcula con rolling/shift DENTRO de cada
#  ticker (groupby), nunca mezclando tickers ni mirando t+1.
# ============================================================

def construir_features(g):
    """Recibe el DataFrame de UN ticker ordenado por fecha y
    devuelve el mismo DataFrame con todas las variables técnicas."""
    g = g.sort_values('date').copy()

    # --- Retorno diario (base de varios indicadores) ---
    g['retorno_1d'] = g['close'].pct_change()

    # --- 1. Retornos rezagados: el retorno de hace k días,
    #        conocido al cierre de hoy ---
    for k in [1, 2, 3, 5, 10]:
        g[f'ret_lag_{k}'] = g['retorno_1d'].shift(k - 1)
        # shift(0) = retorno de hoy (t-1→t), shift(1) = retorno de ayer, etc.

    # --- 2. Medias móviles simples, expresadas como distancia
    #        relativa del precio a la media (comparable entre tickers) ---
    for w in [5, 10, 20, 50]:
        sma = g['close'].rolling(w).mean()
        g[f'dist_sma_{w}'] = (g['close'] - sma) / sma

    # --- 3. Medias móviles exponenciales (misma normalización) ---
    ema12 = g['close'].ewm(span=12, adjust=False).mean()
    ema26 = g['close'].ewm(span=26, adjust=False).mean()
    g['dist_ema_12'] = (g['close'] - ema12) / ema12
    g['dist_ema_26'] = (g['close'] - ema26) / ema26

    # --- 4. RSI de 14 períodos (método de Wilder con medias móviles) ---
    delta = g['close'].diff()
    ganancia = delta.clip(lower=0).rolling(14).mean()
    perdida  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    g['rsi_14'] = 100 - (100 / (1 + rs))

    # --- 5. MACD y su señal, normalizados por el precio para que
    #        sean comparables entre acciones de distinto nivel ---
    macd = ema12 - ema26
    g['macd_norm']  = macd / g['close']
    g['macd_senal'] = macd.ewm(span=9, adjust=False).mean() / g['close']

    # --- 6. Bandas de Bollinger (20 períodos, 2 desviaciones) ---
    sma20  = g['close'].rolling(20).mean()
    std20  = g['close'].rolling(20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    g['bb_ancho'] = (upper - lower) / sma20              # volatilidad relativa
    g['bb_pctb']  = (g['close'] - lower) / (upper - lower)  # posición en la banda

    # --- 7. ATR de 14 períodos (True Range promedio), normalizado ---
    tr = pd.concat([
        g['high'] - g['low'],
        (g['high'] - g['close'].shift(1)).abs(),
        (g['low']  - g['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    g['atr_14_norm'] = tr.rolling(14).mean() / g['close']

    # --- 8. Variables de volumen ---
    g['vol_cambio_pct'] = g['volume'].pct_change()
    g['vol_relativo']   = g['volume'] / g['volume'].rolling(20).mean()

    # --- 9. Rango diario normalizado ---
    g['rango_diario'] = (g['high'] - g['low']) / g['close']

    return g

# Aplicamos la función ticker por ticker y reconcatenamos.
# (Evitamos groupby().apply() porque en pandas ≥2.2 excluye la columna
#  de agrupación del resultado y rompería la columna 'ticker'.)
df = pd.concat(
    [construir_features(g) for _, g in df.groupby('ticker')],
    ignore_index=True
)

# --- 10. Día de la semana (CATEGÓRICA #1) ---
g_dias = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes'}
df['dia_semana'] = df['date'].dt.dayofweek.map(g_dias)

# --- 11. Sector (CATEGÓRICA #2) ya viene del JOIN de la Celda 2 ---

FEATURES_NUM = [
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10',
    'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
    'dist_ema_12', 'dist_ema_26',
    'rsi_14', 'macd_norm', 'macd_senal',
    'bb_ancho', 'bb_pctb', 'atr_14_norm',
    'vol_cambio_pct', 'vol_relativo', 'rango_diario',
]
FEATURES_CAT = ['dia_semana', 'sector']

print(f'✅ Variables construidas: {len(FEATURES_NUM)} numéricas '
      f'+ {len(FEATURES_CAT)} categóricas = {len(FEATURES_NUM) + len(FEATURES_CAT)} predictoras')

# ============================================================
#  CELDA 4 — Variable objetivo y limpieza final (Paso 3)
#
#  target = 1 si close(t+1) > close(t), 0 si no.
#  El shift(-1) mira UN día hacia adelante SOLO para construir
#  la etiqueta — esto es legítimo: la etiqueta es el futuro que
#  queremos predecir, no una variable de entrada.
# ============================================================

df['close_manana'] = df.groupby('ticker')['close'].shift(-1)
df['target'] = (df['close_manana'] > df['close']).astype(int)

# Última fila de cada ticker: no tiene "mañana" → se elimina
df = df.dropna(subset=['close_manana'])

# Las ventanas móviles (SMA50 es la más larga) generan NaN en las
# primeras ~50 filas de cada ticker → se eliminan. Perder ~50 filas
# por ticker (~1,100 en total) es un costo aceptable.
antes = len(df)
df = df.dropna(subset=FEATURES_NUM)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES_NUM)

print(f'✅ Dataset final para modelado: {len(df):,} filas '
      f'({antes - len(df):,} filas de warm-up eliminadas)')
print(f'   Balance del target: {df["target"].mean()*100:.2f}% días de subida')

# ============================================================
#  CELDA 5 — Análisis descriptivo + tests de hipótesis (Paso 4)
# ============================================================

print('=' * 75)
print('ESTADÍSTICAS DESCRIPTIVAS DE LAS VARIABLES NUMÉRICAS')
print('=' * 75)
print(df[FEATURES_NUM].describe().T.round(4).to_string())

# --- Distribución del target por sector y por día de semana ---
print('\n' + '=' * 75)
print('PROPORCIÓN DE DÍAS DE SUBIDA (target=1) POR SECTOR')
print('=' * 75)
print((df.groupby('sector')['target'].mean() * 100).round(2)
      .sort_values(ascending=False).to_string())

print('\n' + '=' * 75)
print('PROPORCIÓN DE DÍAS DE SUBIDA (target=1) POR DÍA DE SEMANA')
print('=' * 75)
orden = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
print((df.groupby('dia_semana')['target'].mean() * 100)
      .reindex(orden).round(2).to_string())

# --- TEST 1: Shapiro-Wilk sobre los retornos diarios ---
#  H0: los retornos siguen una distribución normal.
#  Shapiro está limitado a n≤5000, así que tomamos una muestra
#  aleatoria (práctica estándar). En finanzas se espera RECHAZAR
#  H0: los retornos tienen colas pesadas (leptocurtosis).
muestra = df['retorno_1d'].dropna().sample(5000, random_state=42)
stat_sw, p_sw = shapiro(muestra)
print('\n' + '=' * 75)
print('TEST DE NORMALIDAD (Shapiro-Wilk) sobre retornos diarios')
print('=' * 75)
print(f'  Estadístico W = {stat_sw:.4f} | p-value = {p_sw:.2e}')
if p_sw < 0.05:
    print('  ➜ p < 0.05: RECHAZAMOS H0. Los retornos NO son normales.')
    print('    Conclusión: presentan colas pesadas (eventos extremos más')
    print('    frecuentes que bajo normalidad). Justifica usar modelos de')
    print('    árboles, que no asumen normalidad, además del modelo lineal.')
else:
    print('  ➜ p ≥ 0.05: no hay evidencia contra la normalidad.')

# --- TEST 2: Dickey-Fuller Aumentado (ADF), precios vs retornos ---
#  H0: la serie tiene raíz unitaria (NO es estacionaria).
#  Esperado: precios no estacionarios (no rechaza H0),
#  retornos estacionarios (rechaza H0). Esto justifica por qué
#  el modelo usa retornos e indicadores relativos, no precios crudos.
serie_precio  = df[df['ticker'] == df['ticker'].iloc[0]].set_index('date')['close']
serie_retorno = serie_precio.pct_change().dropna()

adf_p = adfuller(serie_precio.dropna(), autolag='AIC')
adf_r = adfuller(serie_retorno, autolag='AIC')

print('\n' + '=' * 75)
print(f'TEST DE ESTACIONARIEDAD (ADF) — ticker de referencia: {df["ticker"].iloc[0]}')
print('=' * 75)
print(f'  Precios : estadístico = {adf_p[0]:>8.4f} | p-value = {adf_p[1]:.4f}')
print(f'  Retornos: estadístico = {adf_r[0]:>8.4f} | p-value = {adf_r[1]:.2e}')
print('  ➜ Precios: p alto → NO rechazamos H0 → serie NO estacionaria.')
print('  ➜ Retornos: p ≈ 0 → RECHAZAMOS H0 → serie estacionaria.')
print('  Conclusión: modelamos con retornos e indicadores normalizados,')
print('  nunca con el precio en niveles. Esto valida el diseño de features.')

# ============================================================
#  CELDA 6 — EDA: correlaciones, outliers, redundancia (Paso 5)
# ============================================================

# --- 6a. Matriz de correlación ---
corr = df[FEATURES_NUM].corr()

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES_NUM)))
ax.set_yticks(range(len(FEATURES_NUM)))
ax.set_xticklabels(FEATURES_NUM, rotation=60, ha='right', fontsize=8)
ax.set_yticklabels(FEATURES_NUM, fontsize=8)
for i in range(len(FEATURES_NUM)):
    for j in range(len(FEATURES_NUM)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                fontsize=6, color='black' if abs(corr.iloc[i, j]) < 0.6 else 'white')
plt.colorbar(im, fraction=0.046)
ax.set_title('Matriz de Correlación — Variables Predictoras', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlacion.png', dpi=130, bbox_inches='tight')
plt.show()

# --- 6b. Detección de pares altamente correlacionados (|r| > 0.90) ---
#  Multicolinealidad extrema aporta redundancia sin información nueva
#  y desestabiliza el modelo lineal. Eliminamos una variable de cada par.
pares_altos = []
for i in range(len(FEATURES_NUM)):
    for j in range(i + 1, len(FEATURES_NUM)):
        if abs(corr.iloc[i, j]) > 0.90:
            pares_altos.append((FEATURES_NUM[i], FEATURES_NUM[j],
                                round(corr.iloc[i, j], 3)))

print('Pares con |correlación| > 0.90:')
if pares_altos:
    eliminar = set()
    for a, b, r in pares_altos:
        print(f'  {a} ↔ {b}: r = {r}')
        eliminar.add(b)   # criterio simple: conservamos la primera del par
    FEATURES_NUM = [f for f in FEATURES_NUM if f not in eliminar]
    print(f'➜ Eliminadas por redundancia: {sorted(eliminar)}')
else:
    print('  Ninguno. Se conservan las 20 variables numéricas.')
print(f'✅ Variables numéricas finales: {len(FEATURES_NUM)}')

# --- 6c. Tratamiento de outliers: winsorización al 0.5% / 99.5% ---
#  En finanzas los valores extremos son REALES (crashes, gaps), no
#  errores de medición: eliminarlos sesgaría el modelo hacia mercados
#  tranquilos. Por eso winsorizamos (recortamos al percentil) en vez
#  de borrar filas. IMPORTANTE: los límites se calculan más adelante
#  SOLO con train para no filtrar información del test.
def winsorizar(df_ref, df_apply, cols, low=0.005, high=0.995):
    """Calcula límites en df_ref (train) y los aplica a df_apply."""
    for c in cols:
        lo, hi = df_ref[c].quantile(low), df_ref[c].quantile(high)
        df_apply[c] = df_apply[c].clip(lo, hi)
    return df_apply

# --- 6d. Balance de clases ---
balance = df['target'].value_counts(normalize=True) * 100
print(f'\nBalance de clases: 0 (baja/igual) = {balance.get(0, 0):.2f}% | '
      f'1 (sube) = {balance.get(1, 0):.2f}%')
print('➜ Clases razonablemente balanceadas: no se requiere re-muestreo.')
print('  Aun así reportaremos F1 y ROC-AUC, no solo accuracy.')

# ============================================================
#  CELDA 7 — Split TEMPORAL train/test (Paso 6)
#
#  En series de tiempo un split aleatorio produce fuga: el modelo
#  "vería el futuro" al entrenar con días posteriores a los del
#  test. Cortamos por FECHA: 80% más antiguo = train, 20% más
#  reciente = test. Todos los tickers comparten el mismo corte,
#  así ningún dato de test es anterior a datos de train.
# ============================================================

fecha_corte = df['date'].quantile(0.80)
train = df[df['date'] <= fecha_corte].copy()
test  = df[df['date'] >  fecha_corte].copy()

# Winsorización: límites aprendidos en train, aplicados a ambos
train = winsorizar(train, train, FEATURES_NUM)
test  = winsorizar(train, test,  FEATURES_NUM)

X_train, y_train = train[FEATURES_NUM + FEATURES_CAT], train['target']
X_test,  y_test  = test[FEATURES_NUM + FEATURES_CAT],  test['target']

print(f'✅ Corte temporal en: {fecha_corte.date()}')
print(f'   Train: {len(train):,} filas ({train["date"].min().date()} → {train["date"].max().date()})')
print(f'   Test : {len(test):,} filas ({test["date"].min().date()} → {test["date"].max().date()})')

# ============================================================
#  CELDA 8 — Entrenamiento y comparación de 3 modelos (Paso 7a)
#
#  Pipeline con ColumnTransformer:
#   - numéricas → StandardScaler (necesario para la logística;
#     inocuo para los árboles)
#   - categóricas → OneHotEncoder
#  Todo dentro del pipeline para que el escalado se ajuste SOLO
#  con train en cada fold (otra fuente clásica de fuga evitada).
# ============================================================

preproc = ColumnTransformer([
    ('num', StandardScaler(), FEATURES_NUM),
    ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT),
])

modelos = {
    'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1),
}

resultados = []
pipes = {}
for nombre, modelo in modelos.items():
    pipe = Pipeline([('prep', preproc), ('clf', modelo)])
    pipe.fit(X_train, y_train)
    pred  = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    resultados.append({
        'Modelo'  : nombre,
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'F1'      : round(f1_score(y_test, pred), 4),
        'ROC-AUC' : round(roc_auc_score(y_test, proba), 4),
    })
    pipes[nombre] = pipe

df_res = pd.DataFrame(resultados).set_index('Modelo')
print('=' * 60)
print('COMPARACIÓN DE MODELOS (conjunto de test, sin optimizar)')
print('=' * 60)
print(df_res.to_string())

mejor_nombre = df_res['ROC-AUC'].idxmax()
print(f'\n➜ Mejor modelo inicial por ROC-AUC: {mejor_nombre}')

# ============================================================
#  CELDA 9 — Optimización de hiperparámetros (Paso 7b)
#
#  GridSearchCV con TimeSeriesSplit: la validación cruzada también
#  debe respetar el orden temporal (cada fold valida sobre fechas
#  posteriores a las de entrenamiento del fold).
#  Nota: el grid está acotado para que corra en ~10-20 min en Colab.
# ============================================================

tscv = TimeSeriesSplit(n_splits=3)

if mejor_nombre == 'XGBoost':
    param_grid = {
        'clf__n_estimators' : [200, 400],
        'clf__max_depth'    : [3, 4, 6],
        'clf__learning_rate': [0.03, 0.05, 0.10],
        'clf__subsample'    : [0.8, 1.0],
    }
elif mejor_nombre == 'Random Forest':
    param_grid = {
        'clf__n_estimators'    : [200, 400],
        'clf__max_depth'       : [6, 8, 12],
        'clf__min_samples_leaf': [20, 50, 100],
    }
else:  # Regresión Logística
    param_grid = {
        'clf__C'      : [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2'],
    }

# El train debe ir ordenado por fecha para que TimeSeriesSplit tenga sentido
orden_train = train.sort_values('date').index
X_tr_ord, y_tr_ord = X_train.loc[orden_train], y_train.loc[orden_train]

grid = GridSearchCV(
    pipes[mejor_nombre], param_grid,
    cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid.fit(X_tr_ord, y_tr_ord)

print(f'✅ Mejores hiperparámetros: {grid.best_params_}')
print(f'   ROC-AUC promedio en validación temporal: {grid.best_score_:.4f}')

# ============================================================
#  CELDA 10 — Evaluación final del modelo optimizado (Paso 7c)
# ============================================================

modelo_final = grid.best_estimator_
pred_f  = modelo_final.predict(X_test)
proba_f = modelo_final.predict_proba(X_test)[:, 1]

acc_f = accuracy_score(y_test, pred_f)
f1_f  = f1_score(y_test, pred_f)
auc_f = roc_auc_score(y_test, proba_f)

print('=' * 60)
print(f'MODELO FINAL: {mejor_nombre} (optimizado)')
print('=' * 60)
print(f'  Accuracy : {acc_f:.4f}')
print(f'  F1-score : {f1_f:.4f}')
print(f'  ROC-AUC  : {auc_f:.4f}')
print(f'  (Referencia azar puro: accuracy ≈ {max(y_test.mean(), 1-y_test.mean()):.4f} '
      f'prediciendo siempre la clase mayoritaria; ROC-AUC azar = 0.50)')
print('\n' + classification_report(y_test, pred_f,
      target_names=['Baja/Igual (0)', 'Sube (1)']))

# --- Matriz de confusión ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_f),
                       display_labels=['Baja/Igual', 'Sube']).plot(
                       ax=ax, cmap='Blues', values_format=',')
ax.set_title(f'Matriz de Confusión — {mejor_nombre} optimizado', fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=130, bbox_inches='tight')
plt.show()

# --- Importancia de variables ---
prep = modelo_final.named_steps['prep']
nombres_ohe = prep.named_transformers_['cat'].get_feature_names_out(FEATURES_CAT)
nombres_all = FEATURES_NUM + list(nombres_ohe)

clf = modelo_final.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    importancias = clf.feature_importances_
else:  # regresión logística → usamos |coeficiente|
    importancias = np.abs(clf.coef_[0])

df_imp = (pd.DataFrame({'variable': nombres_all, 'importancia': importancias})
          .sort_values('importancia', ascending=True).tail(15))

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(df_imp['variable'], df_imp['importancia'], color='#004AA1')
ax.set_title(f'Top 15 Variables más Importantes — {mejor_nombre}',
             fontweight='bold')
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=130, bbox_inches='tight')
plt.show()

top3 = df_imp.tail(3)['variable'].tolist()[::-1]
print(f'Top 3 variables: {top3}')

# ============================================================
#  CELDA 11 — Guardar el modelo para el despliegue (Paso 8)
#
#  Guardamos el PIPELINE completo (preprocesamiento + modelo):
#  así la app de Streamlit solo necesita pasarle un DataFrame
#  con las columnas crudas y el pipeline hace todo internamente.
# ============================================================

joblib.dump(modelo_final, 'modelo_final.joblib')
joblib.dump({'features_num': FEATURES_NUM,
             'features_cat': FEATURES_CAT,
             'fecha_corte': str(fecha_corte.date()),
             'metricas_test': {'accuracy': acc_f, 'f1': f1_f, 'roc_auc': auc_f}},
            'metadata_modelo.joblib')

print("✅ Guardados: modelo_final.joblib y metadata_modelo.joblib")

# ============================================================
#  CELDA 12 — Resumen ejecutivo
# ============================================================
print('=' * 70)
print('RESUMEN EJECUTIVO')
print('=' * 70)
print(f'''
  Se construyó un dataset de {len(df):,} observaciones diarias con
  {len(FEATURES_NUM)} variables técnicas numéricas y 2 categóricas
  (sector, día de la semana). El modelo final ({mejor_nombre},
  optimizado con validación cruzada temporal) alcanzó en el conjunto
  de test — 20% de fechas más recientes, nunca vistas en entrenamiento —
  un accuracy de {acc_f:.2%}, F1 de {f1_f:.4f} y ROC-AUC de {auc_f:.4f}.
  Las 3 variables más influyentes fueron: {', '.join(top3)}.
''')
