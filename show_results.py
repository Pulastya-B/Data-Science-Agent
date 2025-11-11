"""
Show results of code execution tests
"""

import os
from pathlib import Path
from datetime import datetime

print("\n" + "="*80)
print("📊 CODE EXECUTION RESULTS SUMMARY")
print("="*80)

# Check outputs/code directory
code_dir = Path("./outputs/code")
data_dir = Path("./outputs/data")

if code_dir.exists():
    files = list(code_dir.glob("**/*"))
    files = [f for f in files if f.is_file()]
    
    if files:
        print(f"\n✅ Found {len(files)} file(s) in outputs/code/:\n")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
            size = f.stat().st_size
            mod_time = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  📄 {f.name}")
            print(f"     Size: {size:,} bytes")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Path: {f}")
            print()
    else:
        print("\n❌ No files found in outputs/code/")
else:
    print("\n❌ outputs/code/ directory doesn't exist")

if data_dir.exists():
    files = list(data_dir.glob("**/*"))
    files = [f for f in files if f.is_file() and not f.name.endswith('.csv.dvc')]
    
    if files:
        print(f"\n✅ Found {len(files)} file(s) in outputs/data/:\n")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            size = f.stat().st_size
            mod_time = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  📄 {f.name}")
            print(f"     Size: {size:,} bytes")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()

print("="*80)
print("\n💡 To view generated files:")
print("   - Images (.png): Open with default image viewer")
print("   - HTML files: Open in browser (double-click)")
print("   - CSV/JSON: Open with Excel or text editor")
print("="*80)
