# Glosario de indicadores técnicos — qué mide cada uno, su fórmula y cómo interpretarlo

Este documento explica, uno por uno, los 16 indicadores numéricos y las 2 variables categóricas que usa el modelo. Para cada indicador se incluye: qué mide, su fórmula, qué significa un valor alto vs bajo, y qué encontró el modelo real sobre su importancia (según los coeficientes de la Regresión Logística, tesis sección 7.2).

---

## Grupo 1 — Momentum (retornos rezagados)

**`ret_lag_1`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`, `ret_lag_10`**

**Nombre completo:** Retorno rezagado a k días.

**Qué mide:** El retorno porcentual del precio hace k días respecto al día anterior a ese. Es la forma más simple de capturar "momentum": si el precio venía subiendo o bajando recientemente.

**Fórmula:**
```
retorno(t) = (close(t) - close(t-1)) / close(t-1)
ret_lag_k = retorno(t - k + 1)
```

**Cómo interpretarlo:** Un valor positivo significa que hubo una subida en ese día específico hace k días; negativo, una bajada. No tiene un "rango bueno o malo" universal — su interpretación depende del contexto (¿el modelo interpreta una racha de subidas como algo que continúa, o como algo que va a revertir?).

**Lo que encontró el modelo:** `ret_lag_1` tuvo coeficiente **negativo** (−0.046): una subida ayer *reduce* la probabilidad de que suba hoy. Es el patrón de **reversión a la media de corto plazo** — después de un movimiento brusco, el mercado tiende a corregir en el muy corto plazo.

---

## Grupo 2 — Tendencia (distancia a medias móviles)

**`dist_sma_5`, `dist_sma_10`, `dist_sma_20`, `dist_sma_50`**

**Nombre completo:** Distancia relativa a la Media Móvil Simple (SMA) de N días.

**Qué mide:** Qué tan lejos está el precio de hoy respecto a su propio promedio de los últimos N días, en porcentaje. Es el indicador de tendencia más clásico del análisis técnico.

**Fórmula:**
```
SMA(N) = promedio(close de los últimos N días)
dist_sma_N = (close(t) - SMA(N)) / SMA(N)
```

**Cómo interpretarlo:**
- **Valor positivo** = el precio está por ENCIMA de su media → tendencia alcista reciente (o "sobrecomprado" si es muy alto).
- **Valor negativo** = el precio está por DEBAJO de su media → tendencia bajista reciente (o "sobrevendido" si es muy negativo).
- **Cerca de cero** = el precio está en línea con su tendencia reciente, sin desviación fuerte.

**Regla general de mercado:** valores muy alejados de cero (en cualquier dirección) suelen anticipar una corrección hacia la media — el precio "regresa" a su promedio con el tiempo.

**Lo que encontró el modelo:** aquí está uno de los hallazgos más interesantes de la tesis — `dist_sma_10` (plazo corto) tuvo el coeficiente **negativo** más grande del modelo (−0.064): precio muy estirado sobre su media de 10 días predice una *caída* al día siguiente (reversión a la media). Pero `dist_sma_20` (plazo un poco más largo) tuvo signo **positivo** (+0.042): una tendencia sostenida durante 20 días sí es una señal de continuidad. Es decir, el modelo distingue entre un estirón de muy corto plazo (mala señal) y una tendencia más establecida (buena señal).

---

## Grupo 3 — Osciladores de momentum

**`rsi_14`**

**Nombre completo:** Índice de Fuerza Relativa (Relative Strength Index) de 14 períodos.

**Qué mide:** Qué tan fuertes han sido las subidas comparadas con las bajadas en los últimos 14 días, en una escala de 0 a 100.

**Fórmula:**
```
RS = promedio(ganancias de 14 días) / promedio(pérdidas de 14 días)
RSI = 100 - (100 / (1 + RS))
```

**Cómo interpretarlo (regla clásica de análisis técnico):**
- **RSI > 70** = "sobrecomprado": ha subido mucho y muy rápido, riesgo de corrección a la baja.
- **RSI < 30** = "sobrevendido": ha bajado mucho y muy rápido, riesgo de rebote al alza.
- **RSI ≈ 50** = equilibrio entre compradores y vendedores, sin sesgo fuerte.

**Lo que encontró el modelo:** en el dataset real, el RSI promedió 53.7 — coherente con un mercado que sube ligeramente más de lo que baja en general. Su coeficiente individual fue pequeño (la regularización lo consideró poco informativo una vez que ya están las distancias a SMA), pero sigue siendo estándar reportarlo por su interpretabilidad.

---

## Grupo 4 — Cambios de tendencia (MACD)

**`macd_norm`, `macd_senal`** *(nota: estas dos fueron eliminadas del modelo final por redundancia con las SMA — se explican igual porque aparecen en el código de la Fase 2 y son estándar de la industria)*

**Nombre completo:** Convergencia/Divergencia de Medias Móviles (Moving Average Convergence Divergence) y su línea de señal.

**Qué mide:** La diferencia entre dos medias móviles exponenciales (12 y 26 días), normalizada por el precio. La "señal" es un suavizado del MACD mismo.

**Fórmula:**
```
EMA(N) = media móvil exponencial de N días (da más peso a los días recientes)
MACD = (EMA(12) - EMA(26)) / close
señal = EMA(9) del MACD
```

**Cómo interpretarlo:** cuando el MACD cruza por encima de su señal, se interpreta como una señal alcista; cuando cruza por debajo, bajista. Es un indicador de cambio de tendencia, no de nivel.

**Por qué se eliminó:** se correlacionaba en 0.95 con `dist_sma_50` — es prácticamente la misma información expresada de otra forma. Se mantuvieron las SMA por ser más simples de interpretar.

---

## Grupo 5 — Volatilidad

**`bb_ancho`, `bb_pctb`**

**Nombre completo:** Ancho de las Bandas de Bollinger y Posición Porcentual dentro de ellas (%B).

**Qué mide:** Las Bandas de Bollinger son un canal alrededor del precio, construido con la media móvil de 20 días más/menos 2 desviaciones estándar. El "ancho" mide qué tan volátil ha sido el precio recientemente; el "%B" mide en qué parte de ese canal está el precio ahora.

**Fórmula:**
```
SMA(20), STD(20) = media y desviación estándar de los últimos 20 días
banda_superior = SMA(20) + 2 × STD(20)
banda_inferior = SMA(20) - 2 × STD(20)
bb_ancho = (banda_superior - banda_inferior) / SMA(20)
bb_pctb  = (close - banda_inferior) / (banda_superior - banda_inferior)
```

**Cómo interpretarlo:**
- `bb_ancho` **alto** = mercado muy volátil recientemente (bandas muy separadas); **bajo** = mercado tranquilo, "comprimido" (a veces anticipa un movimiento fuerte próximo, en cualquier dirección).
- `bb_pctb` **cerca de 1** = el precio está pegado a la banda superior (posible sobrecompra); **cerca de 0** = pegado a la banda inferior (posible sobreventa); **≈0.5** = en el centro del canal.

**Lo que encontró el modelo:** `bb_pctb` tuvo coeficiente positivo (+0.039) — estar más cerca de la banda superior se asoció levemente con continuar subiendo, no con revertir, en este dataset.

---

## Grupo 6 — Volatilidad de rango (ATR)

**`atr_14_norm`**

**Nombre completo:** Rango Verdadero Promedio (Average True Range) de 14 períodos, normalizado por el precio.

**Qué mide:** Qué tan amplio ha sido el rango de movimiento diario (considerando gaps de apertura), en promedio, en los últimos 14 días. Es una medida de volatilidad distinta a la de Bollinger porque no depende de la dirección, solo de la magnitud del movimiento.

**Fórmula:**
```
Rango Verdadero(t) = máximo entre:
  high(t) - low(t)
  |high(t) - close(t-1)|
  |low(t) - close(t-1)|
ATR(14) = promedio de los últimos 14 Rangos Verdaderos
atr_14_norm = ATR(14) / close
```

**Cómo interpretarlo:** valores altos = el activo se está moviendo mucho día a día (más riesgo, más oportunidad); valores bajos = movimiento diario contenido. En el dataset real promedió 0.0198 — es decir, en un día típico el precio se mueve cerca de 2% de su valor.

---

## Grupo 7 — Volumen

**`vol_cambio_pct`, `vol_relativo`**

**Nombre completo:** Cambio porcentual del volumen y Volumen relativo a su media de 20 días.

**Qué mide:** Si hoy se negoció una cantidad de acciones inusual comparada con lo normal reciente. Un salto de volumen suele acompañar noticias importantes o cambios de convicción del mercado.

**Fórmula:**
```
vol_cambio_pct = (volumen(t) - volumen(t-1)) / volumen(t-1)
vol_relativo   = volumen(t) / promedio(volumen de los últimos 20 días)
```

**Cómo interpretarlo:** `vol_relativo` > 1 significa que hoy se negoció más de lo normal; < 1, menos de lo normal. Valores muy por encima de 1 (ej. >2) suelen señalar un evento relevante (noticia, resultado financiero, rumor).

---

## Grupo 8 — Rango diario

**`rango_diario`**

**Nombre completo:** Amplitud de la sesión, normalizada por el cierre.

**Qué mide:** Qué tan "movido" estuvo el precio dentro de un mismo día (diferencia entre el máximo y el mínimo).

**Fórmula:**
```
rango_diario = (high(t) - low(t)) / close(t)
```

**Cómo interpretarlo:** valores altos = sesión muy volátil intradía; valores bajos = sesión tranquila.

---

## Variables categóricas

**`sector`** — El sector GICS de la empresa (Technology, Financial Services, Energy, etc.). El modelo usa esta variable porque el desempeño y el comportamiento técnico varían sistemáticamente entre industrias.

**Lo que encontró el modelo:** pertenecer al **Index ETF** (SPY) tuvo el coeficiente positivo más grande de todo el modelo (+0.107) — el sesgo alcista es más fuerte y estable en un índice diversificado que en una acción individual.

**`dia_semana`** — El día de la semana (lunes a viernes). Existe literatura financiera sobre "efectos de calendario" — patrones sistemáticos ligados al día de la semana, mes del año, etc.

**Lo que encontró el modelo:** el **jueves** tuvo coeficiente positivo (+0.062) — en el dataset histórico fue el día con mayor proporción de subidas (53.18%).

---

## Tabla resumen de referencia rápida

| Indicador | Qué mide | Valor alto sugiere | Valor bajo sugiere |
|---|---|---|---|
| ret_lag_k | Momentum de hace k días | Subida reciente | Bajada reciente |
| dist_sma_N | Distancia a la tendencia de N días | Sobrecompra / tendencia alcista | Sobreventa / tendencia bajista |
| rsi_14 | Fuerza relativa 0-100 | Sobrecompra (>70) | Sobreventa (<30) |
| bb_ancho | Volatilidad reciente | Mercado agitado | Mercado tranquilo |
| bb_pctb | Posición en el canal de precio | Cerca del techo | Cerca del piso |
| atr_14_norm | Amplitud de movimiento diario | Alta volatilidad | Baja volatilidad |
| vol_cambio_pct / vol_relativo | Actividad de negociación | Interés/noticia inusual | Sesión tranquila |
| rango_diario | Amplitud de la sesión de hoy | Día muy volátil | Día tranquilo |

**Nota importante sobre "bueno" y "malo":** en análisis técnico clásico estos indicadores tienen lecturas de manual (RSI>70 = "vender", etc.), pero el modelo de este proyecto **no sigue esas reglas de manual** — aprendió sus propios pesos a partir de los datos históricos, y varios de esos pesos (como la reversión a la media de corto plazo en `dist_sma_10`) son más sutiles que la regla clásica. Por eso el valor real de este documento es explicar qué mide cada variable, no prescribir una regla de "compra/venta" — esa interpretación ya está incorporada en los coeficientes del modelo mismo.
