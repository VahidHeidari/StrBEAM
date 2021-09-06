@echo off

python mk_dataset.py 2 50 100 50 &&												^
python -m unittest discover -s Libs -p *.py

