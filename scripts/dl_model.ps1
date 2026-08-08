# Download DINOv2 model files directly via HF mirror
$outDir = "E:\code\subspacead\weights\dinov2"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$base = "https://hf-mirror.com/facebook/dinov2-with-registers-base/resolve/main"
$files = @(
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "model.safetensors"
)

foreach ($f in $files) {
    $url = "$base/$f"
    $out = Join-Path $outDir $f
    Write-Host "Downloading $f ..." -NoNewline
    try {
        Invoke-WebRequest -Uri $url -OutFile $out -TimeoutSec 120 -ErrorAction Stop
        $size = (Get-Item $out).Length / 1MB
        Write-Host " OK ($([math]::Round($size,1)) MB)" -ForegroundColor Green
    } catch {
        Write-Host " SKIPPED (not found or timeout)" -ForegroundColor Yellow
    }
}

Write-Host "`nFiles in $outDir :" -ForegroundColor Cyan
Get-ChildItem $outDir | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB,1)) KB)" }
