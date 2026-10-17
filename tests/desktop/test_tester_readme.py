"""The tester READMEs get their checksums from the files themselves (apps/desktop/scripts/make-tester-readme.sh)."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "apps" / "desktop" / "scripts" / "make-tester-readme.sh"
TEMPLATES = ROOT / "apps" / "desktop" / "tester"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None, reason="needs bash and sha256sum")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args, check=True):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], capture_output=True, text=True, check=check)


def make(tmp_path, template, appimage, extra=()):
    files = tmp_path / "files"
    files.mkdir()
    (files / appimage).write_bytes(b"app image bytes " * 1000)
    for name in extra:
        (files / name).write_bytes(b"zip bytes " * 500)
    out = tmp_path / "README.txt"
    return files, out


def test_every_hash_and_size_in_the_readme_comes_from_the_file(tmp_path):
    files, out = make(tmp_path, "README-linux.txt.in", "Strata-tester-linux.AppImage")
    run(TEMPLATES / "README-linux.txt.in", out, "--appimage", "Strata-tester-linux.AppImage", "--files-dir", files)
    text = out.read_text()
    assert sha(files / "Strata-tester-linux.AppImage") in text
    assert "@" not in text.replace("@", "", 0) or "@APPIMAGE@" not in text
    assert "libfuse2t64" in text and "--browser" in text and "--quit" in text and "Ubuntu 26.04" in text
    assert "Strata opened in your browser because the app window couldn't start on this computer." in text
    run("--verify", out, "--files-dir", files)


@pytest.mark.parametrize("variant", ["rx9070", "rx7900"])
def test_amd_readmes_hash_both_files(tmp_path, variant):
    app, zipname = f"Strata-tester-linux-amd-{variant}.AppImage", f"Strata-Terminal-Linux-tester-amd-{variant}.zip"
    files, out = make(tmp_path, f"README-amd-{variant}.txt.in", app, extra=(zipname,))
    run(TEMPLATES / f"README-amd-{variant}.txt.in", out, "--appimage", app, "--files-dir", files)
    text = out.read_text()
    assert sha(files / app) in text and sha(files / zipname) in text
    assert f"./{app} --browser" in text


def test_a_missing_file_stops_the_readme_instead_of_leaving_a_stale_hash(tmp_path):
    files, out = make(tmp_path, "README-amd-rx9070.txt.in", "Strata-tester-linux-amd-rx9070.AppImage")  # the zip is not there
    result = run(TEMPLATES / "README-amd-rx9070.txt.in", out, "--appimage", "Strata-tester-linux-amd-rx9070.AppImage", "--files-dir", files, check=False)
    assert result.returncode != 0 and "must not carry a hash it did not compute" in result.stderr
    assert not out.exists()


def test_verify_catches_a_stale_hash(tmp_path):
    files, out = make(tmp_path, "README-linux.txt.in", "Strata-tester-linux.AppImage")
    run(TEMPLATES / "README-linux.txt.in", out, "--appimage", "Strata-tester-linux.AppImage", "--files-dir", files)
    (files / "Strata-tester-linux.AppImage").write_bytes(b"rebuilt")  # the file changed after the README was written
    result = run("--verify", out, "--files-dir", files, check=False)
    assert result.returncode != 0 and "STALE" in result.stderr
