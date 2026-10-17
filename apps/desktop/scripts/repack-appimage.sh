#!/usr/bin/env bash
# Put the bundled payloads back to the bytes they were staged with, inside a finished AppImage.
#
#   apps/desktop/scripts/repack-appimage.sh <in.AppImage> <out.AppImage> [--resources <src-tauri/resources>] [--check]
#
# Why: Tauri hands the AppDir to linuxdeploy, which treats every ELF file under usr/lib as a library to fix up: it rewrites the RPATH of
# the engine (strata, libstrata_kq.so, libcudart), of the bundled Python and Node and of the Harness's native modules. The files then no
# longer match the SHA-256 and size their manifests list (engine-component.json), so the host refuses the bundled engine, and the
# installer's own integrity promise ("what we staged is what runs") is broken. This script unpacks the AppImage, copies the staged
# folders (control, harness, node, engine-bundle) over the AppDir byte for byte, and packs it again with the same runtime.
#
# --check does not write anything: it lists every engine-bundle file whose bytes differ from the manifest in the AppImage.
set -euo pipefail

IN="$(realpath "${1:?in.AppImage}")"; OUT="${2:-}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src-tauri/resources" && pwd)"
CHECK=0
shift 2 || true
while [ $# -gt 0 ]; do
  case "$1" in
    --resources) RES="$(cd "$2" && pwd)"; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

OFFSET="$("$IN" --appimage-offset)"
WORK="$(mktemp -d -t strata-repack.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
unsquashfs -no-progress -o "$OFFSET" -d "$WORK/AppDir" "$IN" >/dev/null
# the product folder under usr/lib (Tauri names it after productName)
LIB="$(find "$WORK/AppDir/usr/lib" -maxdepth 2 -type d -name engine-bundle -printf '%h\n' | head -1)"
[ -n "$LIB" ] || LIB="$(find "$WORK/AppDir/usr/lib" -maxdepth 2 -type d -name control -printf '%h\n' | head -1)"
[ -n "$LIB" ] || { echo "no Strata resource folder in the AppImage" >&2; exit 1; }

if [ "$CHECK" = 1 ]; then
  bad=0
  while IFS= read -r f; do
    rel="${f#$RES/}"
    if ! cmp -s "$f" "$LIB/$rel"; then echo "differs: $rel"; bad=$((bad + 1)); fi
  done < <(find "$RES/engine-bundle" "$RES/node" -type f ! -name .gitignore)
  echo "$bad file(s) differ from what was staged"
  [ "$bad" = 0 ]
  exit
fi

[ -n "$OUT" ] || { echo "give out.AppImage" >&2; exit 2; }
for d in control harness node engine-bundle; do
  [ -d "$RES/$d" ] || continue
  rsync -aL --checksum --delete --exclude .gitignore "$RES/$d/" "$LIB/$d/"
done
mksquashfs "$WORK/AppDir" "$WORK/app.squashfs" -root-owned -noappend -comp zstd -b 131072 -no-progress >/dev/null
head -c "$OFFSET" "$IN" > "$OUT"
cat "$WORK/app.squashfs" >> "$OUT"
chmod +x "$OUT"
echo "repacked: $OUT ($(stat -c %s "$OUT") bytes)"
