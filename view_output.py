"""
Quick viewer for generated outputs
"""

import os
import sys
from pathlib import Path

code_dir = Path("./outputs/code")

if not code_dir.exists():
    print("❌ No outputs/code/ directory found")
    sys.exit(1)

files = list(code_dir.glob("**/*"))
files = [f for f in files if f.is_file()]

if not files:
    print("❌ No files found in outputs/code/")
    sys.exit(1)

# Sort by modification time (newest first)
files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

print("\n📁 Generated files (newest first):\n")
for i, f in enumerate(files, 1):
    print(f"{i}. {f.name} ({f.stat().st_size:,} bytes)")

# Auto-open the newest file
newest = files[0]
print(f"\n🚀 Opening: {newest}")

if sys.platform == "win32":
    os.startfile(newest)
elif sys.platform == "darwin":
    os.system(f'open "{newest}"')
else:
    os.system(f'xdg-open "{newest}"')

print("✅ Done!")
