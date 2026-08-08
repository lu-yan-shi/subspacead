import os, sys

# Try multiple mirrors
mirrors = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

os.makedirs("weights/dinov2", exist_ok=True)

for mirror in mirrors:
    os.environ["HF_ENDPOINT"] = mirror
    print(f"Trying mirror: {mirror}")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            "facebook/dinov2-with-registers-base",
            local_dir="weights/dinov2",
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded to: {path}")
        files = os.listdir("weights/dinov2")
        print(f"Files ({len(files)}): {files[:10]}...")
        sys.exit(0)
    except Exception as e:
        print(f"  Failed: {e}")
        print(f"  Trying next mirror...")

print("All mirrors failed. Check your network/VPN.")
sys.exit(1)
