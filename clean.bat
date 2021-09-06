@echo off

REM Clean compiled python files.
del /q /s *.pyc

REM Clean crash artefacts.
del /q /s *.swp *~ gdb.log *.stackdump

REM Uncomment to clean synthetic datasets.
REM del /q /s diffs_* dise_* freqs_* genos_*

REM Clean LaTeX temporary files.
del /q /s *.aux *.bbl *.bcf *.blg *.brf *.gz *.zip *.rar *.log *.out *.pdf ^
*.xml *~ *.synctex *.snm *.toc *.nav

REM Outputs
del /q /s py-beam-posterior.txt py-sum.txt py-site.txt py-intn.txt py-beam3-g.dot ^
smpl_p-* smpl_q-* smpl_z-*

