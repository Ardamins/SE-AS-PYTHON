"""
Configuracion de modelos disponibles y sus clases.

Edita este archivo para anadir/quitar modelos o letras.

IMPORTANTE:
- "input_size" DEBE coincidir con el tamano con el que se entreno el modelo.
- "classes" DEBE estar en el MISMO ORDEN en que se entreno (indice 0 = primera clase).
- Si el numero de clases NO coincide con la salida del modelo TFLite, las
  predicciones seran incorrectas o fallara la inferencia.
"""

# ---------------------------------------------------------------------------
# Definicion de modelos
# ---------------------------------------------------------------------------
# Cada modelo tiene:
#   display_name : nombre legible que aparece en el <select> de la interfaz
#   input_size   : (ancho, alto) que espera el modelo. Ej: (200, 200) o (224, 224)
#   classes      : lista de etiquetas en el orden exacto del entrenamiento
# ---------------------------------------------------------------------------

MODELS = {
    "ResNet50V2_final.tflite": {
        "display_name": "ResNet50V2 (recomendado)",
        "input_size": (160, 160),
        # Letras en el MISMO ORDEN en que se entreno el modelo (indice 0 = 'A', etc.)
        # Sin J ni R (ajustalo a tu dataset real)
        "classes": list("ABCDEFGHIKLMNOPQ"),
    },
    "VGG16_final.tflite": {
        "display_name": "VGG16",
        "input_size": (160, 160),
        "classes": list("ABCDEFGHIKLMNOPQ"),
    },
}

# Modelo que se carga al arrancar el servidor
DEFAULT_MODEL = "ResNet50V2_final.tflite"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def get_model_config(filename: str):
    """
    Devuelve (input_size, classes) para un modelo.

    Si el nombre no existe en MODELS, usa el DEFAULT_MODEL como fallback
    para no romper el servidor.
    """
    cfg = MODELS.get(filename, MODELS[DEFAULT_MODEL])
    return cfg["input_size"], cfg["classes"]


def get_display_name(filename: str) -> str:
    """Devuelve el nombre legible de un modelo (o el filename si no existe)."""
    return MODELS.get(filename, {}).get("display_name", filename)


def validate_models():
    """
    Comprueba que la estructura de MODELS es coherente.
    Lanza ValueError si hay algo mal, para detectar errores al editar.
    """
    if DEFAULT_MODEL not in MODELS:
        raise ValueError(
            f"DEFAULT_MODEL '{DEFAULT_MODEL}' no esta definido en MODELS. "
            f"Disponibles: {list(MODELS.keys())}"
        )

    for name, cfg in MODELS.items():
        if "input_size" not in cfg or "classes" not in cfg:
            raise ValueError(
                f"El modelo '{name}' debe tener 'input_size' y 'classes'."
            )

        size = cfg["input_size"]
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or not all(isinstance(n, int) and n > 0 for n in size)):
            raise ValueError(
                f"'{name}': input_size debe ser una tupla de 2 enteros positivos, "
                f"ej: (200, 200). Recibido: {size!r}"
            )

        classes = cfg["classes"]
        if not isinstance(classes, (list, tuple)) or len(classes) == 0:
            raise ValueError(
                f"'{name}': 'classes' debe ser una lista no vacia."
            )

        if len(set(classes)) != len(classes):
            raise ValueError(
                f"'{name}': hay clases duplicadas en {classes!r}"
            )


# Validar al importar el modulo (falla rapido si hay error de configuracion)
validate_models()