# Download DINOv2 model for SubspaceAD
# Run from PowerShell: .\scripts\download_model.ps1

$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "Downloading DINOv2 model..." -ForegroundColor Cyan

python -c @"
from huggingface_hub import snapshot_download
import os

os.makedirs('weights/dinov2', exist_ok=True)
path = snapshot_download(
    'facebook/dinov2-with-registers-base',
    local_dir='weights/dinov2',
    local_dir_use_symlinks=False,
)
print('Downloaded to:', path)

# Verify
files = os.listdir('weights/dinov2')
print('Files:', files)
for f in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
    if f in files:
        print(f'  [OK] {f}')
    else:
        print(f'  [MISSING] {f}')
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDone! Now restart Docker:" -ForegroundColor Green
    Write-Host "  docker-compose -f docker-compose.yml --profile gpu down" -ForegroundColor Yellow
    Write-Host "  docker-compose -f docker-compose.yml --profile gpu up -d" -ForegroundColor Yellow
} else {
    Write-Host "`nDownload failed. Try:" -ForegroundColor Red
    Write-Host "  1. Check your network" -ForegroundColor Red
    Write-Host "  2. Try VPN or different mirror: `$env:HF_ENDPOINT='https://huggingface.co'" -ForegroundColor Red
}
