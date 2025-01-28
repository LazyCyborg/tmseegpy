# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Extensive PyQt5 exclusions
qt5_excludes = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',
    'PyQt5.QtWebEngine',
    'PyQt5.QtWebEngineCore',
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtNetwork',
    'PyQt5.QtWebChannel',
    'PyQt5.QtWebKit',
    'PyQt5.QtDBus',
    'PyQt5.QtDesigner',
    'PyQt5.QtHelp',
    'PyQt5.QtLocation',
    'PyQt5.QtMultimedia',
    'PyQt5.QtMultimediaWidgets',
    'PyQt5.QtOpenGL',
    'PyQt5.QtPositioning',
    'PyQt5.QtQml',
    'PyQt5.QtQuick',
    'PyQt5.QtQuickWidgets',
    'PyQt5.QtSensors',
    'PyQt5.QtSerialPort',
    'PyQt5.QtSql',
    'PyQt5.QtSvg',
    'PyQt5.QtTest',
    'PyQt5.QtWebSockets',
    'PyQt5.QtXml',
    'PyQt5.QtXmlPatterns'
]

# Collect all required modules
hidden_imports = [
    'tmseegpy',
    'mne',
    'scipy',
    'numpy',
    'pandas',
    'flask',
    'flask_cors',
    'flask_socketio',
    'eventlet',
    'PyQt6',  # We'll use PyQt6 and exclude PyQt5
    'tensorly',
    'matplotlib',
    'seaborn',
    'edfio',
    'eeglabio',
    'antropy',
] + collect_submodules('tmseegpy')

# Collect data files
datas = collect_data_files('tmseegpy')
datas += collect_data_files('mne')

a = Analysis(
    ['tmseegpy/cli.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=qt5_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove any PyQt5 binaries that might have been collected
a.binaries = TOC([x for x in a.binaries if not x[0].startswith('PyQt5')])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='tmseegpy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Server Analysis with the same exclusions
server_a = Analysis(
    ['tmseegpy/server/__init__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=qt5_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove PyQt5 binaries from server analysis as well
server_a.binaries = TOC([x for x in server_a.binaries if not x[0].startswith('PyQt5')])

server_pyz = PYZ(server_a.pure, server_a.zipped_data, cipher=block_cipher)

server_exe = EXE(
    server_pyz,
    server_a.scripts,
    server_a.binaries,
    server_a.zipfiles,
    server_a.datas,
    [],
    name='tmseegpy-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)