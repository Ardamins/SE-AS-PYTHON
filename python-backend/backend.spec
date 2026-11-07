# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None
base_path = Path('.').absolute()

# ---- Datos de la app ----
datas = [
    (str(base_path / 'data'), 'data'),
]

# ---- TensorFlow: datos internos ----
datas += collect_data_files('tensorflow')

# ---- OpenCV: CRÍTICO ----
# cv2 empaqueta DLLs nativas que PyInstaller no detecta con un simple hiddenimport.
# collect_dynamic_libs('cv2') las encuentra automáticamente.
binaries = collect_dynamic_libs('cv2')

# ---- TensorFlow: DLLs nativas ----
binaries += collect_dynamic_libs('tensorflow')

# ---- Hidden imports (solo lo que usamos) ----
hiddenimports = [
    'cv2',
    'numpy',
    'tensorflow',
    'tensorflow.lite',
    'tensorflow.lite.python',
    'tensorflow.lite.python.interpreter',
    'tensorflow.lite.python.schema_py_generated',
]

a = Analysis(
    ['src/main.py'],
    pathex=[str(base_path)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tkinter', 'PyQt5', 'PyQt6',
        'PySide2', 'PySide6', 'IPython', 'jupyter',
        'notebook', 'pandas',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='handsup-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='handsup-backend',
)