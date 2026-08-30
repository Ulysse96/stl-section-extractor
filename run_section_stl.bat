@echo off
py section_stl.py > log.txt 2>&1
type log.txt
echo.
echo ======================================
echo Termine (ou erreur ci-dessus). Le detail complet est aussi dans log.txt.
echo ======================================
pause
