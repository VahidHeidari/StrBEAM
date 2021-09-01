@echo off

REM Build script for making paper from .tex files.

set TEX_FILE=A-10-95-1-Heidari


pdflatex -synctex=1 -halt-on-error -file-line-error %TEX_FILE% &&	^
pdflatex -synctex=1 -halt-on-error -file-line-error %TEX_FILE% &&	^
copy /b %TEX_FILE%.pdf C:\Users\Vahid\Documents\Algorithm\StrBEAM\_Slides\%TEX_FILE%.pdf /b && ^
echo Done! &&														^
start %TEX_FILE%.pdf

