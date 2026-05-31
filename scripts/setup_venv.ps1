# setup_venv.ps1
# FarmTech Solutions — Grupo Aura | Fase 7
# Cria e configura o ambiente virtual Python

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   FarmTech Solutions — Setup Fase 7      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verifica Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python não encontrado. Instale em https://python.org" -ForegroundColor Red
    exit 1
}
$pyVersion = python --version
Write-Host "✅ $pyVersion encontrado." -ForegroundColor Green

# Cria ambiente virtual
Write-Host "`n📦 Criando ambiente virtual (.venv)..." -ForegroundColor Yellow
python -m venv .venv

# Ativa ambiente virtual
Write-Host "⚡ Ativando ambiente virtual..." -ForegroundColor Yellow
try {
    .\.venv\Scripts\Activate.ps1
} catch {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\.venv\Scripts\Activate.ps1
}

# Atualiza pip
Write-Host "`n⬆️  Atualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Instala dependências
Write-Host "`n📥 Instalando dependências (requirements.txt)..." -ForegroundColor Yellow
pip install -r requirements.txt

# Configura .env se não existir
if (-not (Test-Path "config\.env")) {
    Write-Host "`n⚙️  Criando config\.env a partir do exemplo..." -ForegroundColor Yellow
    Copy-Item "config\.env.example" "config\.env"
    Write-Host "   ⚠️  Edite config\.env com suas chaves antes de executar!" -ForegroundColor DarkYellow
} else {
    Write-Host "`n✅ config\.env já existe." -ForegroundColor Green
}

Write-Host "`n╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Setup concluído com sucesso! ✅         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "`nPara iniciar o dashboard, execute:"
Write-Host "   .\scripts\run_dashboard.ps1" -ForegroundColor Cyan
