"""Motor de recomendación que el estudiante debe completar.

Se proporcionan carga, validaciones y funciones auxiliares. Complete únicamente
los cuatro bloques señalados. Las puntuaciones de similitud no son probabilidades.
Este ejercicio no entrena un modelo supervisado: representa textos mediante
TF-IDF y actualiza un perfil con una regla explícita.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def cargar_catalogo(ruta: Path) -> pd.DataFrame:
    """Carga y valida el CSV proporcionado, sin modificar su orden."""
    recursos = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    columnas = ["id", "titulo", "descripcion", "etiquetas"]
    faltantes = set(columnas) - set(recursos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}.")
    if recursos.empty:
        raise ValueError("El catálogo no contiene recursos.")
    for columna in columnas:
        recursos[columna] = recursos[columna].str.strip()
        if (recursos[columna] == "").any():
            raise ValueError(f"La columna {columna} contiene valores vacíos.")
    if recursos["id"].duplicated().any():
        raise ValueError("Los identificadores de recursos deben ser únicos.")
    return recursos.reset_index(drop=True)


def cargar_perfiles(ruta: Path) -> dict:
    """Los intereses vacíos son válidos y se usan en la prueba T4."""
    with ruta.open(encoding="utf-8") as archivo:
        perfiles = json.load(archivo)
    if not isinstance(perfiles, dict) or not perfiles:
        raise ValueError("Se requiere un objeto JSON con perfiles de usuario.")
    for usuario_id, datos in perfiles.items():
        if not isinstance(usuario_id, str) or not usuario_id.strip():
            raise ValueError("Hay un identificador de usuario inválido.")
        if not isinstance(datos, dict):
            raise ValueError(f"El perfil {usuario_id} debe ser un objeto.")
        for campo in ("nombre", "intereses"):
            if not isinstance(datos.get(campo), str):
                raise ValueError(f"El campo {campo} de {usuario_id} debe ser texto.")
    return perfiles


def construir_indice(recursos: pd.DataFrame) -> tuple:
    """El vocabulario se aprende una vez y únicamente a partir de etiquetas."""
    vectorizador = TfidfVectorizer()
    matriz = vectorizador.fit_transform(recursos["etiquetas"]).toarray()
    return vectorizador, matriz


def crear_perfil(intereses: str, vectorizador: TfidfVectorizer) -> np.ndarray:
    """Devuelve un vector de una dimensión con el vocabulario del catálogo."""
    # se usa transform para no cambiar el vocabulario del catalogo
    # si no hay terminos conocidos (u3, u4) queda un vector de ceros
    # toarray crea un array nuevo, asi cada usuario tiene su propio vector
    return vectorizador.transform([intereses]).toarray()[0]


def calcular_puntuaciones(perfil: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Devuelve una puntuación por fila del catálogo, en el mismo orden."""
    # [perfil] porque cosine_similarity pide 2D, y [0] para volver a 1D
    # no redondeamos aca, solo al mostrar
    # con perfil en ceros sklearn devuelve 0 en todo, sin NaN
    return cosine_similarity([perfil], matriz)[0]


def recomendar(
    perfil: np.ndarray,
    recursos: pd.DataFrame,
    matriz: np.ndarray,
    vectorizador: TfidfVectorizer,
    top_k: int = 3,
) -> list[dict]:
    """Devuelve hasta top_k resultados positivos usando construir_resultado."""
    puntuaciones = calcular_puntuaciones(perfil, matriz)
    # solo puntuaciones > 0, no rellenamos con ceros
    candidatos = [i for i, puntuacion in enumerate(puntuaciones) if puntuacion > 0]
    # mayor a menor y si empatan gana el id menor
    candidatos.sort(key=lambda i: (-puntuaciones[i], recursos.iloc[i]["id"]))
    # si el perfil es nulo la lista queda vacia
    return [
        construir_resultado(i, puntuaciones[i], perfil, recursos, matriz, vectorizador)
        for i in candidatos[:top_k]
    ]


# peso del recurso valorado, valor del enunciado (no es experimental)
ALPHA = 0.6


def actualizar_perfil(
    perfil: np.ndarray, vector_recurso: np.ndarray, alpha: float = ALPHA
) -> np.ndarray:
    """Actualiza el perfil cuando el usuario pulsa Me interesa sobre un recurso."""
    # mezcla 0.4 perfil + 0.6 recurso, crea un array nuevo y no modifica el original
    mezcla = (1 - alpha) * np.asarray(perfil, dtype=float) + alpha * np.asarray(
        vector_recurso, dtype=float
    )
    # normalizamos a norma 1 para que el perfil no crezca con cada clic
    return normalize([mezcla])[0]


def obtener_indice(recursos: pd.DataFrame, recurso_id: str) -> int:
    """Valida el identificador antes de modificar un perfil."""
    coincidencias = recursos.index[recursos["id"] == recurso_id].tolist()
    if not coincidencias:
        raise ValueError(f"No existe el recurso {recurso_id}.")
    return int(coincidencias[0])


def explicar_coincidencias(
    perfil: np.ndarray,
    vector_recurso: np.ndarray,
    vectorizador: TfidfVectorizer,
) -> str:
    """La explicación se limita a términos compartidos con peso positivo."""
    vocabulario = vectorizador.get_feature_names_out()
    mascara = (np.asarray(perfil) > 0) & (np.asarray(vector_recurso) > 0)
    terminos = vocabulario[mascara].tolist()
    if not terminos:
        return "No hay términos compartidos con peso positivo."
    return "Términos compartidos: " + ", ".join(terminos) + "."


def construir_resultado(
    indice: int,
    puntuacion: float,
    perfil: np.ndarray,
    recursos: pd.DataFrame,
    matriz: np.ndarray,
    vectorizador: TfidfVectorizer,
) -> dict:
    """Función auxiliar para usar dentro de recomendar."""
    recurso = recursos.iloc[indice]
    return {
        "id": str(recurso["id"]),
        "titulo": str(recurso["titulo"]),
        "puntuacion": float(puntuacion),
        "explicacion": explicar_coincidencias(perfil, matriz[indice], vectorizador),
    }


def describir_perfil(perfil: np.ndarray, vectorizador: TfidfVectorizer) -> dict:
    """Representación legible del estado; no cambia el vector utilizado."""
    vocabulario = vectorizador.get_feature_names_out()
    pares = [(str(termino), float(peso)) for termino, peso in zip(vocabulario, perfil)
             if peso > 0]
    pares.sort(key=lambda par: (-par[1], par[0]))
    return {termino: round(peso, 6) for termino, peso in pares}
