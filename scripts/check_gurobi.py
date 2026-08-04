"""Report whether this machine can actually solve the planning model with Gurobi.

Three things have to line up and each fails differently:

* a license has to be found at all;
* its major version has to cover the installed gurobipy, because a license
  covers its own version and earlier ones but not later ones;
* it has to be unrestricted, since the license bundled with the pip package
  stops at 2000 variables while this model has tens of thousands.

The last check is the decisive one, so it solves a model deliberately larger
than the restricted limit rather than trusting the file's contents.

    python scripts/check_gurobi.py
"""

from __future__ import annotations

import os
from pathlib import Path

PROBE_VARIABLES = 5000  # comfortably past the 2000-variable restricted limit


def find_license_files() -> list[Path]:
    candidates = []
    from_env = os.environ.get("GRB_LICENSE_FILE")
    if from_env:
        candidates.append(Path(from_env))
    # The installation directory is a common place for it and is easy to miss,
    # because GUROBI_HOME points at the platform subdirectory rather than the
    # release root.
    install_root = os.environ.get("GUROBI_HOME")
    if install_root:
        base = Path(install_root)
        candidates += [
            base / "gurobi.lic",
            base / "bin" / "gurobi.lic",
            base.parent / "gurobi.lic",
        ]
    home = Path.home()
    candidates += [
        home / "gurobi.lic",
        home / ".gurobi" / "gurobi.lic",
        Path("C:/gurobi/gurobi.lic"),
        Path("/opt/gurobi/gurobi.lic"),
        Path("/usr/local/lib/gurobi.lic"),
    ]
    for drive in ("C:", "D:", "E:", "F:"):
        for pattern in ("gurobi*/gurobi.lic", "gurobi*/*/gurobi.lic"):
            candidates += list(Path(f"{drive}/").glob(pattern))
    seen, found = set(), []
    for path in candidates:
        try:
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
        except OSError:
            continue
    return found


def describe(path: Path) -> dict[str, str]:
    fields = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                key = key.strip().upper()
                # Key material is not needed to judge the license and should not
                # be echoed to a terminal or a log.
                fields[key] = "<redacted>" if key in {"KEY", "CKEY"} else value.strip()
    except OSError as exc:
        fields["ERROR"] = str(exc)
    return fields


def find_installation() -> list[str]:
    """Standalone Gurobi installs, which carry their own version number."""

    found = []
    home = os.environ.get("GUROBI_HOME")
    if home:
        found.append(f"GUROBI_HOME = {home}")
    for drive in ("C:", "D:", "E:", "F:"):
        for path in Path(f"{drive}/").glob("gurobi*"):
            if path.is_dir():
                found.append(str(path))
    for name in ("gurobi_cl", "gurobi_cl.exe"):
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            if entry and (Path(entry) / name).is_file():
                found.append(f"{name} on PATH at {entry}")
    return found


def run_gurobi_cl() -> str:
    """Ask the installed command-line tool for its version and license state."""

    import shutil
    import subprocess

    executable = shutil.which("gurobi_cl")
    if executable is None:
        install_root = os.environ.get("GUROBI_HOME")
        if install_root:
            for candidate in (
                Path(install_root) / "bin" / "gurobi_cl.exe",
                Path(install_root) / "bin" / "gurobi_cl",
            ):
                if candidate.is_file():
                    executable = str(candidate)
                    break
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(could not run gurobi_cl: {exc})"
    return (completed.stdout + completed.stderr).strip()


def main() -> None:
    print("=" * 68)
    print("Gurobi availability")
    print("=" * 68)

    # The license scan runs first and unconditionally: when gurobipy is missing,
    # the license version is exactly what decides which gurobipy to install.
    print(f"GRB_LICENSE_FILE: {os.environ.get('GRB_LICENSE_FILE', '(not set)')}")
    licenses = find_license_files()
    print(f"license files found: {len(licenses)}")
    license_versions = []
    for path in licenses:
        fields = describe(path)
        version = fields.get("VERSION", "?")
        if version.isdigit():
            license_versions.append(int(version))
        print(f"\n  {path}")
        for key in ("TYPE", "VERSION", "EXPIRATION", "HOSTNAME", "HOSTID", "CORES"):
            if key in fields:
                print(f"    {key:<11} {fields[key]}")

    installations = find_installation()
    if installations:
        print("\nstandalone installs:")
        for entry in installations:
            print(f"  {entry}")

    # gurobi_cl is the authority on both the installed version and whether the
    # license it can see actually works, so ask it rather than inferring.
    banner = run_gurobi_cl()
    if banner:
        print("\ngurobi_cl --version:")
        for line in banner.splitlines():
            if line.strip():
                print(f"  {line.strip()}")

    print()
    try:
        import gurobipy as gp
    except ImportError:
        print("gurobipy    : NOT INSTALLED")
        if license_versions:
            newest = max(license_versions)
            print(f"\nThe license above covers Gurobi {newest}. A license covers its")
            print("own version and earlier ones, never later ones, so install:")
            print(f"    python -m pip install \"gurobipy=={newest}.*\"")
        else:
            print("\nNo license file was found either. Request a free academic license")
            print("at https://www.gurobi.com/academia/, activate it with grbgetkey on")
            print("the university network, then install a matching gurobipy.")
        return

    library_version = gp.gurobi.version()
    print(f"gurobipy    : {'.'.join(map(str, library_version))}")
    for version in license_versions:
        if version < library_version[0]:
            print(f"  -> MISMATCH: a license for Gurobi {version} cannot run gurobipy "
                  f"{library_version[0]}.")
            print(f"     python -m pip install \"gurobipy=={version}.*\"")

    print("\n" + "-" * 68)
    print(f"solving a {PROBE_VARIABLES}-variable model to test the real limit")
    print("-" * 68)
    try:
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        variables = model.addVars(PROBE_VARIABLES, vtype=gp.GRB.BINARY)
        model.setObjective(gp.quicksum(variables.values()))
        model.addConstr(gp.quicksum(variables.values()) >= PROBE_VARIABLES // 2)
        model.optimize()
        print(f"RESULT: usable. Solved {PROBE_VARIABLES} binaries, objective "
              f"{model.ObjVal:.0f}.")
        print("\nSet `solver_backend: gurobi` under `planning:` in the config.")
    except gp.GurobiError as exc:
        print(f"RESULT: NOT usable -- {exc}")
        print("\nThe planning model has tens of thousands of variables, so a")
        print("size-limited license cannot run it. Request a free academic")
        print("license at https://www.gurobi.com/academia/ and activate it with")
        print("grbgetkey while on the university network.")


if __name__ == "__main__":
    main()
