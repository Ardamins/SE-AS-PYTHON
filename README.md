# 🤟 HandsUp — Detector de Lenguaje de Señas

Aplicación de escritorio (**Electron + Python**) que reconoce letras del alfabeto
de señas a partir de una imagen, usando un modelo **TensorFlow Lite** ejecutado
en un backend local.

---

## 📋 Requisitos previos

Antes de empezar, asegúrate de tener instalado:

| Herramienta | Versión recomendada | Cómo comprobar |
|---|---|---|
| **Python** | **3.10.x** (obligatorio para TensorFlow 2.13) | `py -3.10 --version` |
| **Node.js** | LTS 18 o 20 | `node --version` |
| **npm** | Incluido con Node.js | `npm --version` |
| **Git** | Opcional (para clonar) | `git --version` |

> ⚠️ **Muy importante**: NO uses Python 3.11, 3.12, 3.13 ni 3.14.
> `tensorflow==2.13.0` solo tiene paquetes precompilados para **Python 3.10**
> (y 3.8–3.11). Con otra versión, `pip install -r requirements.txt` fallará.

---

## 📁 Estructura del proyecto
