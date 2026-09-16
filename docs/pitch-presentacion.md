# Pitch — BTC News Signal (5 min + demo en vivo)

> Ensayar en voz alta: ~650 palabras ≈ 5 minutos. Marcas de tiempo entre bloques.
> Idioma: español neutro de exposición. Demo: `cd web; python -m http.server 8000` → http://localhost:8000

## 0. Setup previo (antes de entrar, 2 min)

- [ ] Servidor levantado y pestaña abierta en la web (verificar que cargan las 4 curvas).
- [ ] Terminal lista con el CLI por si piden veredicto en vivo:
  `btc-signal --title "Bitcoin ETF inflows hit record" --price 60000`
- [ ] Plan B: si falla el wifi/proyector, este mismo guion funciona contando las
  cifras de memoria (están todas abajo). Sin localhost no hay demo, pero sí historia.

## 1. Gancho — 30 s

«El ejercicio pedía un clasificador de compra, vende o mantén basado en noticias
de Bitcoin. Yo me hice una pregunta incómoda antes de entrenar nada: **¿y si las
noticias no predicen el precio?** Spoiler: tenía razón. Y demostrarlo con
protocolo serio es este proyecto.»

## 2. Qué construí — 45 s

«Un pipeline honesto de punta a punta: noticias de Kaggle cruzadas con precio
diario de Kraken —siempre la vela *pasada*, nunca la futura—, split cronológico
con purga y embargo, y el test virgen leído **una sola vez**. Cuatro motores
compitiendo en el mismo split: baseline TF-IDF, LLM zero-shot, extractor de
embeddings y un fine-tune LoRA entrenado en GPU en Kaggle. Todo reproducible:
84 tests verdes, tag v1.0 en GitHub.»

## 3. Demo en vivo — 90 s

*(Abrir la web. Recorrer en este orden.)*

1. **Velas + señales**: «Cada vela es un día del test; los triángulos son lo que
   votó cada motor ese día.»
2. **Curvas de capital**: desmarcar todo menos HODL y baseline. «El testigo
   HODL —comprar el día 1 y no tocar— hace 1.32x. El baseline, 0.89x.»
3. **El momento clave**: activar solo la curva rosa del LoRA. «Plana en 1.00x.
   No es un bug: el modelo predijo *sell* 999 de 1000 veces sin tener Bitcoin
   nunca, así que se quedó en cash todo el año. El colapso se ve de un vistazo.»
4. **Simulador**: cambiar el monto a 5000 €. «Todo reescala; la conclusión no
   cambia con el monto.»

## 4. El resultado incómodo — 60 s

«Macro-F1 en test virgen: extractor 0.368, baseline 0.336, zero-shot 0.283,
LoRA 0.197 —con el azar en 0.33—. Los cuatro viven pegados al azar. Y en
capital: HODL 1.32 > extractor 1.17 > LoRA 1.00 > LLM 0.92 > baseline 0.89.
Moraleja con números: **con señales flojas, tradear pierde contra no hacer
nada**. El LoRA fue un negativo honesto: hiperparámetros congelados antes de
ver el test, cero tuneo, documentado como D11.»

## 5. Cierre — 45 s

«Me llevo tres cosas. Una: el proceso vale más que el modelo —cualquiera puede
mostrar un 0.90 con leakage; mostrar un 0.20 honesto es lo difícil. Dos: para
clasificar, cabezal de clasificación le gana a generación de texto. Tres: el
pipeline quedó reutilizable —repetir el experimento es apretar un botón—.
Repo público, versión 1.0, gracias.»

## Preguntas probables (y respuestas de 15 s)

- **¿Por qué no más épocas de LoRA?** «El rango 0.20–0.37 contra azar 0.33 dice
  que la señal es flojísima; más GPU no cambia la historia y quema cupo.»
- **¿No habrás tuneado con el test?** «Test-once con protocolo: split
  cronológico + purga + embargo, y el LoRA se evaluó en el complemento disjunto
  del test —cero overlap con la muestra de la web.»
- **¿Por qué el extractor gana?** «Embeddings congelados + regresión simple:
  clasifica en vez de generar texto, y no se inventa atajos como el LoRA.»
- **¿Esto sirve para tradear?** «No. Es educativo —la propia web lo dice— y su
  conclusión es no tradear con esto.»
