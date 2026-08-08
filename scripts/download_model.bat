@echo off
REM Download DINOv2 model for SubspaceAD
set HF_ENDPOINT=https://hf-mirror.com

echo Downloading DINOv2 model...
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-with-registers-base', local_dir='weights/dinov2', local_dir_use_symlinks=False); print('Done')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Verified files:
    dir weights\dinov2
    echo.
    echo Now restart Docker:
    echo   docker-compose -f docker-compose.yml --profile gpu down
    echo   docker-compose -f docker-compose.yml --profile gpu up -d
) else (
    echo Download failed. Try setting VPN or changing mirror:
    echo   set HF_ENDPOINT=https://huggingface.co
    echo Then re-run this script.
)
pause
