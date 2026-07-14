# ¿Se puede predecir la dirección del mercado accionario?

Proyecto Final — Diplomado en Data Science & Machine Learning (4Geeks Academy)
**Autor:** Raúl Márquez

Clasificación binaria de la dirección diaria del precio (sube/baja mañana) usando
indicadores técnicos, sobre 82,896 registros de 22 tickers (2011–2026) adquiridos
vía la API pública de Yahoo Finance y gobernados en SQLite.

## Resultados en una línea

El modelo óptimo (Regresión Logística, C=0.01, seleccionada por validación cruzada
temporal) alcanzó **51.80% de accuracy y ROC-AUC 0.4961** sobre 16,346 días fuera de
muestra — desempeño en el nivel del azar, consistente con la hipótesis de mercados
eficientes en horizonte diario. El análisis granular muestra que la predictibilidad
se concentra en índices y large caps estables (SPY 57.1%) y que las señales aportan
como filtro de riesgo en activos débiles (PFE: de −15.0% a +3.2% en backtest).

## Estructura del repositorio

```
├── 01_adquisicion_datos.py     # Fase 1: API yfinance → SQLite (SELECT/JOIN/INSERT)
├── 02_features_eda_modelo.py   # Fase 2: 22 features, tests, EDA, 3 modelos, GridSearch
├── Proyecto_Raul_Marquez.ipynb # Notebook con la ejecución completa y salidas
├── app.py                      # Fase 3: aplicación web Streamlit
├── modelo_final.joblib         # Pipeline entrenado (preprocesamiento + modelo)
├── requirements.txt
├── docs/
│   ├── Tesis_Proyecto_Final.pdf        # Informe completo (16 págs) con Q&A
│   └── Presentacion_Proyecto_Final.pptx
└── README.md
```

## Cómo reproducir

1. **Datos y base:** `python 01_adquisicion_datos.py` genera `mercado.db` (~83k filas).
2. **EDA y modelo:** `python 02_features_eda_modelo.py` entrena, optimiza y guarda
   `modelo_final.joblib` (tarda 15–30 min por el GridSearch).
3. **App en local:** `pip install -r requirements.txt` y `streamlit run app.py`.

## Cómo desplegar en Streamlit Community Cloud (gratis)

1. Sube este repositorio a GitHub (público). **Deben estar en la raíz:** `app.py`,
   `requirements.txt` y `modelo_final.joblib` (pesa ~360 KB, entra sin problema).
2. Entra a **share.streamlit.io** e inicia sesión con tu cuenta de GitHub.
3. "Create app" → elige el repositorio, rama `main`, archivo principal `app.py`.
4. Deploy. La primera construcción tarda 2–4 minutos (instala scikit-learn 1.6.1).
5. La URL pública resultante (`https://<tu-app>.streamlit.app`) es el enlace del
   entregable.

**Nota crítica:** `requirements.txt` fija `scikit-learn==1.6.1` porque el modelo
fue serializado con esa versión; con otra versión el `joblib.load` falla.

## Advertencia

Modelo académico con desempeño cercano al azar, backtest sin costos de transacción.
**Esto no constituye una recomendación de inversión.**
