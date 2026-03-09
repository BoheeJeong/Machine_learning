@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [LAB 05] 폴더에서 TabPy를 시작합니다.
echo 이 창을 닫지 마세요. Tableau에서 연결 후 사용하세요.
echo.
tabpy
pause
