# hac_exmp

# Eco Platform API (MVP)

FastAPI + PostgreSQL backend для "Эко-платформы": пользователи, транзакции с расчетом углеродного следа, челленджи/баллы, точки приема.

## Стек
Python 3.10+, FastAPI, SQLAlchemy 2, PostgreSQL, asyncpg, PyJWT, passlib, Pydantic v2, Uvicorn.

## Запуск локально (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# создаем БД (из psql или любой GUI)
CREATE DATABASE eco_platform;

# переменные окружения на сессию
$env:DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/eco_platform"
$env:JWT_SECRET="dev-secret-change-me"
$env:ADMIN_EMAILS="admin@example.com"

uvicorn eco_platform_fastapi_app:app --reload
