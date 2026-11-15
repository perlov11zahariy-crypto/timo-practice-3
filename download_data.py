"""Загрузка официального архива; SHA-256 распакованного CSV проверяется в train."""
from pathlib import Path
from urllib.request import urlretrieve
root=Path(__file__).resolve().parent
(root/'data').mkdir(exist_ok=True)
urlretrieve('https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip',root/'data/metropt.zip')
print('Архив загружен. Далее: python pipeline.py train')
