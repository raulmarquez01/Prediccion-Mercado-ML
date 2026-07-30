# ¿Se puede predecir la dirección del mercado accionario?

Proyecto Final — Diplomado en Data Science & Machine Learning (4Geeks Academy)
**Autor:** Raúl Márquez

Clasificación binaria de la dirección diaria del precio (sube/baja mañana) usando
indicadores técnicos, sobre un universo de **60 tickers** (2011–2026) en 11 sectores
GICS, clasificados además por estilo de inversión (**Growth / Value / Index**),
adquiridos vía la API pública de Yahoo Finance y gobernados en SQLite.

> **Nota de versión:** este proyecto se amplió tras una asesoría con el tutor del
> programa. La versión inicial (22 tickers, un solo horizonte de 1 día) evolucionó
> a un universo de 60 tickers con clasificación Growth/Value, y está en desarrollo
> la extensión a predicción multi-horizonte (1 semana / 1 mes / 3 meses / 6 meses)
> y construcción de cartera con monto de inversión — ver sección "Roadmap" abajo.

## Resultados en una línea (versión de 22 tickers, línea base)

El modelo óptimo (Regresión Logística, C=0.01, seleccionada por validación cruzada
temporal) alcanzó **51.80% de accuracy y ROC-AUC 0.4961** sobre 16,346 días fuera de
muestra — desempeño en el nivel del azar, consistente con la hipótesis de mercados
eficientes en horizonte diario. El análisis granular muestra que la predictibilidad
se concentra en índices y large caps estables (SPY 57.1%) y que las señales aportan
como filtro de riesgo en activos débiles (PFE: de −15.0% a +3.2% en backtest).

## Estructura del repositorio

```
├── 01_adquisicion_datos.py       # v1: Fase 1 original — 22 tickers
├── 01_universo_ampliado.py       # v2: Fase A — 60 tickers + Growth/Value/Index
├── 02_features_eda_modelo.py     # Fase 2: 22 features, tests, EDA, 3 modelos, GridSearch
├── Proyecto_Raul_Marquez.ipynb   # Notebook con la ejecución completa y salidas
├── mercado.db                    # Base de datos SQLite (universo ampliado, 60 tickers)
├── app.py                        # Fase 3: aplicación web Streamlit
├── modelo_final.joblib           # Pipeline entrenado (preprocesamiento + modelo)
├── glosario_indicadores.md       # Fase D: nombre, fórmula e interpretación de cada indicador
├── requirements.txt
├── runtime.txt                   # Fija Python 3.11 para compatibilidad con scikit-learn 1.6.1
├── docs/
│   ├── Tesis_Proyecto_Final.pdf         # Informe completo (16 págs) con Q&A
│   └── Presentacion_Proyecto_Final.pptx
└── README.md
```

## Cómo reproducir

1. **Datos y base (universo ampliado):** `python 01_universo_ampliado.py` genera
   `mercado.db` (~226k filas, 60 tickers, clasificación Growth/Value/Index).
2. **EDA y modelo:** `python 02_features_eda_modelo.py` entrena, optimiza y guarda
   `modelo_final.joblib` (tarda 15–30 min por el GridSearch).
3. **App en local:** `pip install -r requirements.txt` y `streamlit run app.py`.

## Cómo desplegar en Streamlit Community Cloud (gratis)

1. Sube este repositorio a GitHub (público). **Deben estar en la raíz:** `app.py`,
   `requirements.txt`, `runtime.txt` y `modelo_final.joblib`.
2. Entra a **share.streamlit.io** e inicia sesión con tu cuenta de GitHub.
3. "Create app" → elige el repositorio, rama `main`, archivo principal `app.py`.
4. Deploy. La primera construcción tarda 2–4 minutos.
5. La URL pública resultante es el enlace del entregable.

**Nota crítica:** `requirements.txt` fija `scikit-learn==1.6.1` (versión con la que
se serializó el modelo) y `runtime.txt` fija Python 3.11 — versiones más nuevas de
Python (3.14) no siempre tienen binarios precompilados de scikit-learn y pueden
colgar el deploy.

## Metodología de clasificación Growth / Value (Fase A)

Para cada ticker se obtienen dos ratios fundamentales públicos vía yfinance: P/E
(precio/utilidad) y P/B (precio/valor en libros). Se clasifica como **Growth** si
ambos ratios están por encima de la mediana del universo de 60 acciones; como
**Value** si están por debajo. Los P/B negativos (patrimonio contable negativo, por
recompras de acciones o deuda alta — no significan "barata") se tratan como dato
faltante y la clasificación cae de vuelta al P/E. El índice SPY se etiqueta aparte
como **"Index"** y se excluye de la comparación Growth/Value, ya que un ETF
diversificado no es conceptualmente ni growth ni value.

## Roadmap (trabajo en curso tras asesoría con el tutor)

- [x] Fase A — Universo ampliado a 60 tickers + clasificación Growth/Value/Index
- [x] Fase D — Glosario completo de indicadores (nombre, fórmula, interpretación)
- [ ] Fase B — Modelo multi-horizonte (1 semana / 1 mes / 3 meses / 6 meses) con
      predicción de dirección **y** rendimiento % esperado
- [ ] Fase C — Construcción de cartera: selección de varias acciones, monto de
      inversión hipotético, rendimiento nominal esperado al final del período
- [ ] Fase E — Insights adicionales: comparación Growth vs Value, mejor horizonte
      de inversión, indicadores más relevantes por horizonte

## Advertencia

Modelo académico con desempeño cercano al azar en el horizonte diario, backtest sin
costos de transacción. **Esto no constituye una recomendación de inversión.**

