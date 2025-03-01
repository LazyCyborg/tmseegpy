# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

block_cipher = None

# Collect all necessary data files
streamlit_data = collect_data_files('streamlit')
mne_data = collect_data_files('mne')
tmseegpy_data = collect_data_files('tmseegpy')

# Additional hidden imports
hidden_imports = [
    'streamlit',
    'mne',
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'plotly',
    'PyQt6',
    'tensorly',
    'sklearn',
] + collect_submodules('streamlit')

a = Analysis(
    [os.path.join(current_dir, 'tmseegpy', 'main_gui', 'main.py')],
    pathex=[current_dir],
    binaries=[],
    datas=[
        *streamlit_data,
        *mne_data,
        *tmseegpy_data,

        ('/Users/alexe/Kaggle/tmseegpy/icon/icon.icns', '.')
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HePoTEP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='/Users/alexe/Kaggle/tmseegpy/icon/icon.icns'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HePoTEP'
)

app = BUNDLE(
    coll,
    name='HePoTEP.app',
    icon='/Users/alexe/Kaggle/tmseegpy/icon/icon.icns',
    bundle_identifier='com.hepotep',
    info_plist={
        'CFBundleName': 'HePoTEP',
        'CFBundleDisplayName': 'HePoTEP',
        'CFBundleExecutable': 'HePoTEP',
        'CFBundlePackageType': 'APPL',
        'CFBundleSupportedPlatforms': ['MacOSX'],
        'LSMinimumSystemVersion': '10.13.0',
        'NSHighResolutionCapable': True,
    },
)