#!/usr/bin/env python3
"""
Build a macOS .app bundle for Code Horde Command Center.

Creates:
  dist/Code Horde.app/
    Contents/
      Info.plist
      MacOS/
        Code Horde       ← shell launcher
      Resources/
        icon.icns       ← app icon (generated)
        AppIcon.png     ← source icon

Usage:
    python scripts/build_app.py               # build the .app
    python scripts/build_app.py --install     # build + copy to /Applications
"""

import argparse
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
APP_NAME = "Code Horde"
APP_BUNDLE = os.path.join(DIST_DIR, f"{APP_NAME}.app")
BUNDLE_ID = "dev.agentarmy.commandcenter"
VERSION = "1.0.0"


def create_icon(resources_dir: str) -> str:
    """Generate a simple app icon (cyan shield on dark background).

    Returns the path to the .icns file.
    """
    png_path = os.path.join(resources_dir, "AppIcon.png")
    icns_path = os.path.join(resources_dir, "icon.icns")
    iconset_dir = os.path.join(resources_dir, "icon.iconset")

    # Create a 512x512 PNG icon using Python (no ImageMagick needed)
    try:
        # Try with Pillow first
        from PIL import Image, ImageDraw, ImageFont

        size = 512
        img = Image.new("RGBA", (size, size), (15, 23, 42, 255))  # slate-950
        draw = ImageDraw.Draw(img)

        # Draw a rounded rectangle background
        margin = 40
        draw.rounded_rectangle(
            [margin, margin, size - margin, size - margin],
            radius=60,
            fill=(6, 182, 212, 40),    # brand-500/15
            outline=(6, 182, 212, 120), # brand-500/50
            width=3,
        )

        # Draw a shield shape
        cx, cy = size // 2, size // 2 - 10
        shield_w, shield_h = 140, 170
        points = [
            (cx, cy - shield_h // 2),                     # top
            (cx + shield_w // 2, cy - shield_h // 3),     # top-right
            (cx + shield_w // 2, cy + shield_h // 6),     # mid-right
            (cx, cy + shield_h // 2),                     # bottom
            (cx - shield_w // 2, cy + shield_h // 6),     # mid-left
            (cx - shield_w // 2, cy - shield_h // 3),     # top-left
        ]
        draw.polygon(points, fill=(6, 182, 212, 200), outline=(6, 182, 212, 255))

        # Draw "AA" text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
        except Exception:
            font = ImageFont.load_default()
        draw.text((cx, cy + 5), "AA", fill=(15, 23, 42, 255), font=font, anchor="mm")

        # Draw "AGENT ARMY" below
        try:
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except Exception:
            small_font = ImageFont.load_default()
        draw.text(
            (cx, size - margin - 40), "AGENT ARMY",
            fill=(6, 182, 212, 200), font=small_font, anchor="mm",
        )

        img.save(png_path, "PNG")
        print(f"  Icon PNG created: {png_path}")

    except ImportError:
        # Fallback: create a minimal 1x1 PNG (iconutil will still work)
        print("  Pillow not installed — using placeholder icon")
        # Minimal valid PNG (1x1 cyan pixel)
        import struct
        import zlib

        def make_png(width, height, color):
            """Create a minimal PNG."""

            def chunk(chunk_type, data):
                c = chunk_type + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

            raw = b""
            for _ in range(height):
                raw += b"\x00"  # filter byte
                for _ in range(width):
                    raw += bytes(color)

            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw))
                + chunk(b"IEND", b"")
            )

        with open(png_path, "wb") as f:
            f.write(make_png(512, 512, (6, 182, 212, 255)))

    # Convert PNG → icns using macOS iconutil
    os.makedirs(iconset_dir, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512]
    for s in sizes:
        # Use sips (macOS built-in) to resize
        out = os.path.join(iconset_dir, f"icon_{s}x{s}.png")
        subprocess.run(
            ["sips", "-z", str(s), str(s), png_path, "--out", out],
            capture_output=True,
        )
        # @2x variant
        if s <= 256:
            out2x = os.path.join(iconset_dir, f"icon_{s}x{s}@2x.png")
            subprocess.run(
                ["sips", "-z", str(s * 2), str(s * 2), png_path, "--out", out2x],
                capture_output=True,
            )

    result = subprocess.run(
        ["iconutil", "-c", "icns", iconset_dir, "-o", icns_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  Icon .icns created: {icns_path}")
    else:
        print(f"  Warning: iconutil failed ({result.stderr.strip()}), icon may be missing")

    # Cleanup iconset
    shutil.rmtree(iconset_dir, ignore_errors=True)

    return icns_path


def create_launcher(macos_dir: str) -> str:
    """Create the shell script that launches the Python app."""
    launcher_path = os.path.join(macos_dir, APP_NAME)

    # Resolve the Python binary at build time so we can embed it as a fallback
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    resolved_python = ""
    if os.path.exists(venv_python):
        resolved_python = os.path.realpath(venv_python)
        print(f"  Build-time Python resolved: {resolved_python}")

    # Resolve paths relative to the .app bundle
    # NOTE: we write a plain string (no f-string) so shell $(...) is not
    # mangled by Python.  Only PROJECT_ROOT / RESOLVED_PYTHON are injected.
    script = textwrap.dedent("""\
        #!/bin/bash
        # Code Horde — macOS app launcher

        # Log file for debugging (visible via Console.app or tail)
        LOG="$HOME/Library/Logs/Code Horde.log"
        exec >> "$LOG" 2>&1
        echo ""
        echo "=== Code Horde launch $(date) ==="

        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

        # ── Force native architecture on Apple Silicon ──
        # If this .app is accidentally launched under Rosetta, re-exec natively.
        HW_ARCH=$(sysctl -n hw.optional.arm64 2>/dev/null)
        CURR_ARCH=$(uname -m)
        if [ "$HW_ARCH" = "1" ] && [ "$CURR_ARCH" = "x86_64" ]; then
            echo "Detected Rosetta — re-launching as arm64..."
            exec arch -arm64 "$0" "$@"
        fi

        # Project root baked in at build time
        PROJECT_ROOT="__PROJECT_ROOT__"

        # Fallback: .app sits inside dist/ inside project
        if [ ! -d "$PROJECT_ROOT/src" ]; then
            PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
        fi
        if [ ! -d "$PROJECT_ROOT/src" ]; then
            PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
        fi

        echo "PROJECT_ROOT=$PROJECT_ROOT"

        # ── Python resolution with architecture validation ──
        SYS_ARCH=$(uname -m)
        VENV="$PROJECT_ROOT/.venv"
        PYTHON=""

        check_python_arch() {
            # Validate that a Python binary matches the current architecture.
            # Returns 0 (success) if compatible, 1 if not.
            local candidate="$1"
            [ ! -x "$candidate" ] && return 1

            # Resolve symlinks to get the real binary (macOS readlink has no -f)
            local resolved
            resolved=$(python3 -c "import os; print(os.path.realpath('$candidate'))" 2>/dev/null || realpath "$candidate" 2>/dev/null || echo "$candidate")
            local file_info
            file_info=$(file "$resolved" 2>/dev/null)

            echo "  Checking $candidate -> $resolved"
            echo "    file: $file_info"

            # Universal binary is always OK
            echo "$file_info" | grep -q "universal" && return 0

            # Match specific architecture
            if [ "$SYS_ARCH" = "arm64" ]; then
                echo "$file_info" | grep -q "arm64" && return 0
            else
                echo "$file_info" | grep -q "x86_64" && return 0
            fi
            return 1
        }

        echo "System arch: $SYS_ARCH"

        # Priority 1: venv Python
        if [ -x "$VENV/bin/python" ]; then
            if check_python_arch "$VENV/bin/python"; then
                PYTHON="$VENV/bin/python"
                echo "Using venv Python: $PYTHON"
            else
                echo "WARNING: venv Python architecture mismatch ($SYS_ARCH)"
            fi
        fi

        # Priority 2: Homebrew Python (arm64 native on Apple Silicon)
        if [ -z "$PYTHON" ] && [ "$SYS_ARCH" = "arm64" ]; then
            for brew_py in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3; do
                if [ -x "$brew_py" ]; then
                    if check_python_arch "$brew_py"; then
                        # Use Homebrew Python but with the venv's site-packages
                        PYTHON="$brew_py"
                        export VIRTUAL_ENV="$VENV"
                        export PYTHONPATH="$VENV/lib/python3.12/site-packages:$VENV/lib/python3.13/site-packages:$PROJECT_ROOT"
                        echo "Using Homebrew Python: $PYTHON (with venv site-packages)"
                        break
                    fi
                fi
            done
        fi

        # Priority 3: Build-time resolved Python path
        BUILD_PYTHON="__RESOLVED_PYTHON__"
        if [ -z "$PYTHON" ] && [ -n "$BUILD_PYTHON" ] && [ -x "$BUILD_PYTHON" ]; then
            if check_python_arch "$BUILD_PYTHON"; then
                PYTHON="$BUILD_PYTHON"
                export VIRTUAL_ENV="$VENV"
                export PYTHONPATH="$VENV/lib/python3.12/site-packages:$VENV/lib/python3.13/site-packages:$PROJECT_ROOT"
                echo "Using build-time Python: $PYTHON"
            fi
        fi

        if [ -z "$PYTHON" ]; then
            echo "ERROR: No compatible Python ($SYS_ARCH) found"
            echo "  Checked: $VENV/bin/python, Homebrew, build-time path"
            osascript -e "display dialog \\"No compatible Python found for $SYS_ARCH.\\n\\nFix: recreate the virtualenv:\\n  rm -rf .venv && make venv deps\\" with title \\"Code Horde\\" buttons {\\"OK\\"} default button \\"OK\\" with icon caution"
            exit 1
        fi

        echo "PYTHON=$PYTHON"

        # Check pywebview is installed
        if ! "$PYTHON" -c "import webview" 2>/dev/null; then
            echo "ERROR: pywebview not installed"
            osascript -e "display dialog \\"pywebview is not installed.\\n\\nRun:  pip install pywebview\\" with title \\"Code Horde\\" buttons {\\"OK\\"} default button \\"OK\\" with icon caution"
            exit 1
        fi

        # Load environment
        if [ -f "$PROJECT_ROOT/.env" ]; then
            set -a
            source "$PROJECT_ROOT/.env"
            set +a
        fi

        cd "$PROJECT_ROOT"
        echo "Launching: $PYTHON -m src.desktop.app --standalone"
        "$PYTHON" -m src.desktop.app --standalone

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: app exited with code $EXIT_CODE"
            osascript -e "display dialog \\"Code Horde crashed (exit code $EXIT_CODE).\\n\\nCheck log: $LOG\\" with title \\"Code Horde\\" buttons {\\"OK\\"} default button \\"OK\\" with icon caution"
        fi
    """).replace("__PROJECT_ROOT__", PROJECT_ROOT
    ).replace("__RESOLVED_PYTHON__", resolved_python)

    with open(launcher_path, "w") as f:
        f.write(script)

    # Make executable
    st = os.stat(launcher_path)
    os.chmod(launcher_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"  Launcher created: {launcher_path}")
    return launcher_path


def create_info_plist(contents_dir: str, icon_name: str) -> str:
    """Create the Info.plist."""
    plist_path = os.path.join(contents_dir, "Info.plist")

    info = {
        "CFBundleDisplayName": "Code Horde",
        "CFBundleName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": icon_name,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSAppTransportSecurity": {
            "NSAllowsLocalNetworking": True,
        },
        # Prefer arm64 (Apple Silicon native) over x86_64 (Rosetta)
        "LSArchitecturePriority": ["arm64", "x86_64"],
    }

    with open(plist_path, "wb") as f:
        plistlib.dump(info, f)

    print(f"  Info.plist created: {plist_path}")
    return plist_path


def build_app() -> str:
    """Build the complete .app bundle."""
    print(f"\nBuilding {APP_NAME}.app ...\n")

    # Clean previous build
    if os.path.exists(APP_BUNDLE):
        shutil.rmtree(APP_BUNDLE)

    # Create directory structure
    contents_dir = os.path.join(APP_BUNDLE, "Contents")
    macos_dir = os.path.join(contents_dir, "MacOS")
    resources_dir = os.path.join(contents_dir, "Resources")

    os.makedirs(macos_dir)
    os.makedirs(resources_dir)

    # Create components
    icns_path = create_icon(resources_dir)
    icon_name = os.path.basename(icns_path) if os.path.exists(icns_path) else "icon.icns"
    create_info_plist(contents_dir, icon_name)
    create_launcher(macos_dir)

    # Remove macOS quarantine flag so Gatekeeper doesn't block it
    subprocess.run(["xattr", "-cr", APP_BUNDLE], capture_output=True)
    print(f"  Quarantine flag removed (xattr -cr)")

    print(f"\n  {APP_NAME}.app built at: {APP_BUNDLE}")
    return APP_BUNDLE


def install_app(app_path: str) -> None:
    """Copy .app to /Applications."""
    dest = f"/Applications/{APP_NAME}.app"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(app_path, dest)
    print(f"  Installed to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME}.app")
    parser.add_argument("--install", action="store_true", help="Copy to /Applications")
    args = parser.parse_args()

    app_path = build_app()

    if args.install:
        install_app(app_path)

    print(f"\nDone! Open with:")
    print(f"  open {app_path}")
    print()


if __name__ == "__main__":
    main()
