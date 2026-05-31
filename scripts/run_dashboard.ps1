# run_dashboard.ps1
# FarmTech Solutions — Grupo Aura | Fase 7
# Inicia o dashboard Streamlit

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   FarmTech Solutions — Iniciando...      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Ativa ambiente virtual se existir
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "⚡ Ativando ambiente virtual..." -ForegroundColor Yellow
    try {
        .\.venv\Scripts\Activate.ps1
    } catch {
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
        .\.venv\Scripts\Activate.ps1
    }
}

# Carrega variáveis de ambiente
if (Test-Path "config\.env") {
    Write-Host "⚙️  Carregando variáveis de ambiente..." -ForegroundColor Yellow
    Get-Content "config\.env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.+)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
} else {
    Write-Host "⚠️  config\.env não encontrado. Execute setup_venv.ps1 primeiro." -ForegroundColor DarkYellow
}

Write-Host "`n🌱 Iniciando FarmTech Dashboard..." -ForegroundColor Green
Write-Host "   Acesse em: http://localhost:8501" -ForegroundColor Cyan
Write-Host "   Pressione Ctrl+C para encerrar.`n"

# Inicia o Streamlit
streamlit run src/dashboard.py --server.port 8501 --server.headless false
