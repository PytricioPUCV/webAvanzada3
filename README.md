# Laboratorio 3: Aplicación inteligente adaptativa

Ingeniería Web Avanzada (OII436-1), segundo semestre 2026.

Integrantes: Patricio Hernandez Vergara, Ignacio Reyes Torres.

Repositorio: https://github.com/PytricioPUCV/webAvanzada3

Recomendador basado en contenido con TF-IDF y similitud del coseno. El perfil del usuario se actualiza al marcar "Me interesa" en un recurso.

---

## Entorno y ejecución

Python 3.11.0 en Windows. Activación usada: PowerShell.

```
python -m venv .venv
```

```
# PowerShell
.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat

# macOS o Linux
source .venv/bin/activate
```

```
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

```
python app.py --comprobar
python app.py --demo
python app.py
```

--comprobar valida los datos y construye la matriz TF-IDF (12 recursos, 4 usuarios, 12 x 17). --demo ejecuta T1 a T4 y guarda los archivos en evidencias/. Sin argumentos se abre el modo interactivo (r recomendar, c catálogo, m Me interesa, u cambiar usuario, s salir). requirements.txt se generó con pip freeze dentro del entorno.

---

## Catálogo y perfiles

Solo la columna etiquetas de datos/recursos.csv interviene en el cálculo. titulo y descripcion son de presentación; id se usa para validar interacciones y desempatar.

cargar_catalogo lanza ValueError si faltan columnas, si hay valores vacíos (tras aplicar strip) o si hay IDs repetidos. app.py muestra el error y termina sin procesar. No se rellenan ni eliminan filas.

En datos/perfiles.json, u3 tiene intereses vacíos y u4 tiene "zzxyz123", fuera del vocabulario. Ambos producen un vector de ceros. Cada usuario tiene su propio vector en estado y su propia lista en historial.

---

## Implementación de motor.py

crear_perfil: vectorizador.transform([intereses]).toarray()[0].

calcular_puntuaciones: cosine_similarity([perfil], matriz)[0], sin redondear. Con perfil nulo devuelve ceros, sin NaN.

recomendar: filtra puntuaciones mayores a 0, ordena por (-puntuación, id) y devuelve hasta 3 resultados con construir_resultado.

actualizar_perfil: normalize([(1 - alpha) * perfil + alpha * vector_recurso])[0], con ALPHA = 0.6 como parámetro opcional. No modifica los vectores de entrada.

motor.py no contiene IDs ni resultados esperados. Las comprobaciones con valores fijos (u1_primero_R02, u2_primero_R07, aumenta_puntuacion_R05, R04_pasa_al_primer_lugar) pertenecen al material base en app.py, que no fue modificado.

---

## Preguntas

1. El contenido de los recursos está en la columna etiquetas de datos/recursos.csv, convertida en la matriz TF-IDF por construir_indice. El perfil parte del campo intereses de datos/perfiles.json, transformado por crear_perfil; durante la ejecución corresponde al vector en estado[usuario_id] y a los eventos en historial[usuario_id].

2. fit_transform aprende el vocabulario y los pesos IDF del catálogo una sola vez; transform proyecta los intereses en ese mismo espacio de 17 términos. Con un vectorizador por perfil, cada vector tendría otras dimensiones y otro orden de términos, por lo que el coseno con la matriz fallaría o compararía términos distintos.

3. Los perfiles apuntan a términos distintos (u1: pandas, datos, python; u2: componentes, interfaces, angular), por lo que el coseno con cada recurso cambia. Una similitud de 0,80 no es una probabilidad: mide coincidencia de términos entre vectores, no está calibrada con preferencias reales y solo sirve para ordenar.

4. Editar los intereses en el JSON es un cambio manual que requiere reiniciar. En la adaptación, el evento es "Me interesa" sobre un recurso; se modifica solo el vector del usuario activo (estado["u1"]) mediante actualizar_perfil y se agrega el evento a su historial. El efecto se observa en la misma ejecución: cambian las puntuaciones y el ranking (R02, R12, R03 pasa a R04, R02, R09).

5. Monitorizar: registrar_interes recibe el ID, lo valida con obtener_indice y lo registra en historial. Analizar: actualizar_perfil combina el perfil con el vector del recurso. Planificar: calcular_puntuaciones y recomendar generan el nuevo ranking. Ejecutar: estado[usuario_id] = nuevo y mostrar_resultados. Conocimiento: datos/, vectorizador, matriz TF-IDF, estado e historial.

6. Solo se usan señales positivas y, con alpha = 0.6, cada interacción desplaza mucho el perfil (R02 bajó de 1.0000 a 0.6880 sin rechazo del usuario). Además, el estado se pierde al reiniciar. Mejora: agregar "No me interesa" que reste el vector del recurso y usar un alpha decreciente, por ejemplo 1/(n+1). Requiere un nuevo tipo de evento en el historial y persistirlo en archivo o base de datos.

---

## Pruebas T1 a T4

| Caso | Entrada | Esperado | Observado | Estado |
|---|---|---|---|---|
| T1 | Recomendar para u1 y u2 | Hasta 3 resultados positivos, orden descendente, desempate por ID | u1: R02 1.0000, R12 0.4965, R03 0.4326. u2: R07 1.0000, R08 0.5677, R09 0.3087 | Cumple |
| T2 | u1 marca Me interesa en R04 | Cambia el vector de u1 y al menos un recurso no valorado, con 12 candidatos | Perfil incorpora api y web. R05: 0.0000 a 0.4378. R09: 0.0000 a 0.4958. Ranking: R04 0.8752, R02 0.6880, R09 0.4958. Historial con evento R04 | Cumple |
| T3 | Consultar u2 tras la interacción de u1 | Vector, historial y puntuaciones sin cambios | Perfil, historial vacío, puntuaciones y ranking idénticos | Cumple |
| T4 | Recomendar para u3 y u4 | Lista vacía y mensaje, sin errores | Puntuaciones en 0, lista vacía, mensaje "No hay coincidencias positivas. Declare intereses presentes en el catálogo." | Cumple |

Pruebas adicionales: el ID R99 muestra "No existe el recurso R99." y no modifica estado ni historial. Los empates se ordenan por ID: R10 antes que R11 (0.2053) y, tras R04, R05 antes que R06 (0.4378).

---

## Rankings iniciales

u1 ("python datos pandas"):

| Puesto | id | Título | Puntuación | Explicación |
|---|---|---|---|---|
| 1 | R02 | Análisis de datos con Pandas | 1.0000 | Términos compartidos: datos, pandas, python. |
| 2 | R12 | Conectar Python con datos SQL | 0.4965 | Términos compartidos: datos, python. |
| 3 | R03 | Gráficos con Python | 0.4326 | Términos compartidos: datos, python. |

u2 ("angular componentes interfaces"):

| Puesto | id | Título | Puntuación | Explicación |
|---|---|---|---|---|
| 1 | R07 | Componentes e interfaces en Angular | 1.0000 | Términos compartidos: angular, componentes, interfaces. |
| 2 | R08 | Formularios e interfaces en Angular | 0.5677 | Términos compartidos: angular, interfaces. |
| 3 | R09 | Consumir una API desde Angular | 0.3087 | Términos compartidos: angular. |

---

## Adaptación de u1

R04 no estaba en el top 3 inicial (4.º lugar, 0.2510) y se valoró desde el catálogo completo.

Perfil antes: pandas 0.753247, datos 0.465091, python 0.465091.
Perfil después: python 0.637116, api 0.446269, web 0.446269, pandas 0.376477, datos 0.232455.
Historial después: {"numero": 1, "accion": "Me interesa", "recurso": "R04"}.

| id | Título | Antes | Después | Cambio |
|---|---|---|---|---|
| R01 | Fundamentos de Python | 0.1861 | 0.2549 | +0.0688 |
| R02 | Análisis de datos con Pandas | 1.0000 | 0.6880 | -0.3120 |
| R03 | Gráficos con Python | 0.4326 | 0.4044 | -0.0282 |
| R04 | API web con Python | 0.2510 | 0.8752 | +0.6241 |
| R05 | Principios de una API HTTP | 0.0000 | 0.4378 | +0.4378 |
| R06 | Validación de datos en una API | 0.0000 | 0.4378 | +0.4378 |
| R07 | Componentes e interfaces en Angular | 0.0000 | 0.0000 | 0.0000 |
| R08 | Formularios e interfaces en Angular | 0.0000 | 0.0000 | 0.0000 |
| R09 | Consumir una API desde Angular | 0.0000 | 0.4958 | +0.4958 |
| R10 | Consultas SQL sobre datos | 0.2053 | 0.1026 | -0.1027 |
| R11 | Tablas de datos en SQL | 0.2053 | 0.1026 | -0.1027 |
| R12 | Conectar Python con datos SQL | 0.4965 | 0.4642 | -0.0324 |

R05 y R09 no compartían términos con el perfil inicial, por lo que su puntuación era 0. Tras valorar R04, el perfil incorporó api y web (0.446269 cada uno). Como los vectores tienen norma 1, el coseno equivale al producto punto: R05 = 2 x 0.446269 x 0.490485 = 0.4378 y R09 = 2 x 0.446269 x 0.555449 = 0.4958.

R02 baja de 1.0000 a 0.6880 porque la mezcla reduce el peso de pandas y datos y el perfil se orienta hacia api y web, términos que R02 no tiene.

---

## u2, u3 y u4

u2 no cambia: perfil componentes 0.657512, interfaces 0.564679, angular 0.498813; historial vacío; ranking R07, R08, R09 con las mismas puntuaciones.

u3 y u4 reciben lista vacía y el mensaje "No hay coincidencias positivas. Declare intereses presentes en el catálogo.", sin errores.

---

## Problemas detectados

No hubo fallos en las pruebas. Se eliminó la carpeta de metadatos de macOS incluida en el zip, se fijaron las versiones en requirements.txt con pip freeze y se agregaron .env, .vscode/ e .idea/ al .gitignore.

---

## Archivos de la entrega

motor.py, app.py, datos/recursos.csv, datos/perfiles.json, evidencias/reporte_demo.json, evidencias/puntuaciones_antes_despues.csv, README.md, requirements.txt y .gitignore. No se incluyen el entorno virtual ni archivos de caché.
