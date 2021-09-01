
REM Clean compiled python files.
del /s *.pyc

REM Uncomment to clean synthetic datasets.
REM del /s genos_* diffs_* freqs_* smpl_*

REM Clean LaTeX temporary files.
del /q /s *.aux *.bbl *.bcf *.blg *.brf *.gz *.zip *.rar *.log *.out *.pdf ^
*.xml *~ *.pyc *.synctex *.snm *.toc *.nav

