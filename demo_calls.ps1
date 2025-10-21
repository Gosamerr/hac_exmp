# ================================
# Eco Platform — demo_calls.ps1
# ================================
$ErrorActionPreference = "Stop"

# Настройки
$BASE = "http://127.0.0.1:8000"
$EMAIL = "loikliza4@gmail.com"
$PASSWORD = "secret"
$NAME = "Liza"

# Файлы логов/ответов
$CLIENT_LOG = "client_run.log"      # читаемый лог действий
$RESP_LOG   = "responses.jsonl"     # «сырые» ответы API (по строке JSON)

if (Test-Path $CLIENT_LOG) { Remove-Item $CLIENT_LOG -Force }
if (Test-Path $RESP_LOG)   { Remove-Item $RESP_LOG   -Force }
Start-Transcript -Path $CLIENT_LOG | Out-Null

function Write-Resp {
    param([string]$label, $data)
    $line = @{ ts = (Get-Date).ToString("o"); step = $label; resp = $data } | ConvertTo-Json -Depth 20
    Add-Content -Path $RESP_LOG -Value $line
}

function Call-Api {
    param(
        [string]$Method,[string]$Url,
        [hashtable]$Headers = @{},
        [string]$ContentType = "",
        $Body = $null,[string]$Label = ""
    )
    Write-Host ">> $Method $Url $Label"
    try {
        $resp = if ($Body -ne $null -and $ContentType) {
            Invoke-RestMethod -Method $Method -Uri $Url -Headers $Headers -ContentType $ContentType -Body $Body
        } elseif ($Body -ne $null) {
            Invoke-RestMethod -Method $Method -Uri $Url -Headers $Headers -Body $Body
        } else {
            Invoke-RestMethod -Method $Method -Uri $Url -Headers $Headers
        }
        Write-Resp $Label $resp
        return $resp
    } catch {
        $err = $_.ErrorDetails.Message; if (-not $err) { $err = $_.Exception.Message }
        Write-Warning "Request failed: $err"
        Write-Resp "$Label(FAILED)" @{ error = $err }
        throw
    }
}

# 0) Health
$health = Call-Api -Method GET -Url "$BASE/health" -Label "health"

# 1) Регистрация → если уже есть — логин
$regBody = @{ email=$EMAIL; password=$PASSWORD; name=$NAME } | ConvertTo-Json
try {
    $reg = Call-Api -Method POST -Url "$BASE/auth/register" -ContentType 'application/json' -Body $regBody -Label "auth_register"
    $TOKEN = $reg.access_token
} catch {
    $loginBody = "username=$($EMAIL)&password=$($PASSWORD)"
    $login = Call-Api -Method POST -Url "$BASE/auth/login" -ContentType 'application/x-www-form-urlencoded' -Body $loginBody -Label "auth_login"
    $TOKEN = $login.access_token
}
$AUTH = @{ Authorization = "Bearer $TOKEN" }

# 2) Профиль
$me = Call-Api -Method GET -Url "$BASE/auth/me" -Headers $AUTH -Label "auth_me"

# 3) Категории
$cats = Call-Api -Method GET -Url "$BASE/categories" -Label "categories_list"

# 4) Создать транзакцию
$txBody = @{
    category_key = "groceries"
    amount      = 1200
    currency    = "RUB"
    metadata    = @{ merchant = "Magnit"; city = "Rostov" }
} | ConvertTo-Json
$tx = Call-Api -Method POST -Url "$BASE/transactions" -Headers $AUTH -ContentType 'application/json' -Body $txBody -Label "tx_create"

# 5) Список транзакций
$txs = Call-Api -Method GET -Url "$BASE/transactions?limit=20&offset=0" -Headers $AUTH -Label "tx_list"

# 6) Сводка
$sum = Call-Api -Method GET -Url "$BASE/transactions/summary" -Headers $AUTH -Label "tx_summary"

# 7) (Опц.) Админ: создать челлендж
try {
    $chBody = @{ title="Пешком на работу 3 дня"; description="3 км в день"; points_reward=50 } | ConvertTo-Json
    $ch  = Call-Api -Method POST -Url "$BASE/challenges" -Headers $AUTH -ContentType 'application/json' -Body $chBody -Label "challenge_create"
    $cid = $ch.id
} catch { }

# 8) Присоединиться/завершить челлендж
if ($cid) {
    Call-Api -Method POST -Url "$BASE/challenges/$cid/join" -Headers $AUTH -Label "challenge_join" | Out-Null
    Call-Api -Method POST -Url "$BASE/challenges/$cid/complete" -Headers $AUTH -Label "challenge_complete" | Out-Null
}

# 9) Точки приёма (лист)
$rp = Call-Api -Method GET -Url "$BASE/recycling_points" -Label "recycling_list"

# 10) (Optional) create recycling point (admin)
try {
    $rpBody = @{ title="Recycling Center"; description="Plastic and glass"; lat=47.231; lon=39.715; tags=@("plastic","glass") } | ConvertTo-Json
    Call-Api -Method POST -Url "$BASE/recycling_points" -Headers $AUTH -ContentType 'application/json' -Body $rpBody -Label "recycling_create" | Out-Null
} catch { }

Write-Host "Done! Logs: $CLIENT_LOG, responses: $RESP_LOG"
Stop-Transcript | Out-Null
