@echo off
title EcoTrack Vehicle CO2 REST API Server
echo ========================================================
echo    🌿 EcoTrack Vehicle CO2 Emissions REST API Server
echo ========================================================
echo.
echo Starting FastAPI with Uvicorn on http://localhost:8000 ...
echo Swagger UI documentation: http://localhost:8000/docs
echo ReDoc documentation:      http://localhost:8000/redoc
echo Web Application:          http://localhost:8000/
echo.
uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload
pause
