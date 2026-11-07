import logging
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import base64
import numpy as np
import cv2
import tensorflow as tf

from .model_config import MODELS, DEFAULT_MODEL, get_model_config

INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = []
CURRENT_MODEL = None
CURRENT_INPUT_SIZE = (200, 200)
MODELS_DIR = None
STATIC_DIR = None


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
def _load_interpreter(model_filename: str):
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    global CURRENT_MODEL, CURRENT_INPUT_SIZE

    if model_filename not in MODELS:
        raise ValueError(f"Modelo desconocido: {model_filename}")

    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el modelo: {path}")

    logging.info(f"Cargando modelo TFLite: {path}")
    INTERPRETER = tf.lite.Interpreter(model_path=path)
    INTERPRETER.allocate_tensors()
    INPUT_DETAILS = INTERPRETER.get_input_details()
    OUTPUT_DETAILS = INTERPRETER.get_output_details()

    size, classes = get_model_config(model_filename)
    CURRENT_INPUT_SIZE = size
    CLASS_NAMES = classes
    CURRENT_MODEL = model_filename
    logging.info(
        f"Modelo cargado: {model_filename} | "
        f"Clases: {len(CLASS_NAMES)} | Input: {size}"
    )


# ---------------------------------------------------------------------------
# Imagenes de ejemplo disponibles en data/static/images
# ---------------------------------------------------------------------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif', '.bmp'}


def _list_sample_images():
    """Devuelve nombres de archivo de las imagenes de ejemplo."""
    if not STATIC_DIR:
        return []
    images_dir = os.path.join(STATIC_DIR, 'images')
    if not os.path.isdir(images_dir):
        return []

    result = []
    for name in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Handler HTTP
# ---------------------------------------------------------------------------
class PredictionHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Permite que <img crossorigin="anonymous"> y canvas.toDataURL()
        # funcionen sin "tainted canvas" al cargar imagenes desde /static.
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    # ----------------------- GET -----------------------
    def do_GET(self):
        # /health
        if self.path == '/health':
            self._set_headers()
            response = {
                'status': 'ok',
                'current_model': CURRENT_MODEL,
                'model': 'loaded' if INTERPRETER else 'not_loaded',
                'classes': CLASS_NAMES,
                'input_size': list(CURRENT_INPUT_SIZE),
            }
            self.wfile.write(json.dumps(response).encode())
            return

        # /models
        if self.path == '/models':
            self._set_headers()
            response = {
                'available': [
                    {'file': k, 'name': v['display_name']}
                    for k, v in MODELS.items()
                ],
                'current': CURRENT_MODEL,
            }
            self.wfile.write(json.dumps(response).encode())
            return

        # /samples -> lista de imagenes de ejemplo
        if self.path == '/samples':
            self._set_headers()
            files = _list_sample_images()
            response = {
                'images': [
                    {'file': f, 'url': f'/static/images/{f}'}
                    for f in files
                ]
            }
            self.wfile.write(json.dumps(response).encode())
            return

        # /static/... -> sirve archivos estaticos (imagenes, etc.)
        if self.path.startswith('/static/'):
            self._serve_static()
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def _serve_static(self):
        if not STATIC_DIR:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Static dir not configured'}).encode())
            return

        rel = self.path[len('/static/'):]

        # Quitar query string (?t=123...) antes de buscar el archivo.
        if '?' in rel:
            rel = rel.split('?', 1)[0]
        # Decodificar %20, %C3%A9, etc. (nombres con acentos)
        try:
            from urllib.parse import unquote
            rel = unquote(rel)
        except Exception:
            pass

        safe_rel = os.path.normpath(rel).replace('\\', '/')
        if safe_rel.startswith('..') or os.path.isabs(safe_rel):
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid path'}).encode())
            return

        full = os.path.join(STATIC_DIR, *safe_rel.split('/'))
        if not os.path.isfile(full):
            logging.warning(f"Estatico no encontrado: {full}")
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'File not found'}).encode())
            return

        ext = os.path.splitext(full)[1].lower()
        mime = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp', '.avif': 'image/avif',
            '.bmp': 'image/bmp',
        }.get(ext, 'application/octet-stream')

        try:
            with open(full, 'rb') as f:
                data = f.read()
            self._set_headers(200, content_type=mime)
            self.wfile.write(data)
        except Exception as e:
            logging.error(f"Error sirviendo {full}: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    # ----------------------- POST -----------------------
    def do_POST(self):
        # /select-model
        if self.path == '/select-model':
            try:
                length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(length))
                _load_interpreter(data.get('model'))
                self._set_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'current_model': CURRENT_MODEL,
                    'classes': CLASS_NAMES,
                }).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # /predict
        if self.path == '/predict':
            try:
                length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(length))

                # Validar que llego la imagen
                if 'image' not in data or not data['image']:
                    raise ValueError("Falta el campo 'image' en la peticion")

                raw = data['image']
                if ',' in raw:
                    raw = raw.split(',', 1)[1]
                image_bytes = base64.b64decode(raw)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is None:
                    raise ValueError("No se pudo decodificar la imagen")

                w, h = CURRENT_INPUT_SIZE
                image = cv2.resize(image, (w, h))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                image = np.expand_dims(image, axis=0).astype(np.float32)

                INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], image)
                INTERPRETER.invoke()
                predictions = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

                # Si el modelo devuelve logits en vez de softmax -> aplicar softmax
                if predictions.sum() < 0.99 or predictions.sum() > 1.01:
                    e = np.exp(predictions - np.max(predictions))
                    predictions = e / e.sum()

                idx = int(np.argmax(predictions))
                confidence = float(predictions[idx])

                probs = {
                    (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"):
                        float(predictions[i])
                    for i in range(len(predictions))
                }

                self._set_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'prediction': CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}",
                    'confidence': confidence,
                    'probabilities': probs,
                    'model': CURRENT_MODEL,
                }).encode())

            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
def start_server(model_path, static_path, port, models_dir=None):
    global MODELS_DIR, STATIC_DIR
    MODELS_DIR = models_dir or os.path.dirname(model_path)
    STATIC_DIR = static_path

    default_file = os.path.basename(model_path)
    if default_file not in MODELS:
        default_file = DEFAULT_MODEL
    _load_interpreter(default_file)

    logging.info(f"Servidor iniciado en http://127.0.0.1:{port}")
    logging.info(f"   Modelos disponibles: {list(MODELS.keys())}")
    logging.info(f"   Imagenes de ejemplo: {len(_list_sample_images())}")

    server = HTTPServer(('127.0.0.1', port), PredictionHandler)
    server.serve_forever()