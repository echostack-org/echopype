import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SUITE = Path(__file__).resolve().parent
REPO = SUITE.parent.parent


def execute(args, **kwargs):
    return subprocess.run(
        [str(arg) for arg in args],
        check=True,
        **kwargs,
    )


def git(*args, cwd=REPO):
    return execute(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkout(ref, destination):
    try:
        commit = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        git("fetch", "origin", ref)
        commit = git("rev-parse", "FETCH_HEAD")

    git("clone", "--no-hardlinks", REPO, destination)
    git("fetch", REPO, commit, cwd=destination)
    git("checkout", "--detach", "FETCH_HEAD", cwd=destination)

    return commit


def run(base, head, benchmark, output, config_path=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    data = output / "data"
    suite = output / "suite"

    config_path = Path(config_path or SUITE / "config.yaml").resolve()

    config = yaml.safe_load(config_path.read_text())

    file_counts = config["scaling"]["file_counts"]
    shutil.copytree(SUITE, suite)

    for params in config["datasets"].values():
        if "manifest" in params:
            params["manifest"] = str((config_path.parent / params["manifest"]).resolve())

    (suite / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    shutil.copy(suite / "config.yaml", output)

    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BENCHMARK_DATA": str(data),
    }

    sources = {}

    for name, ref in {"base": base, "head": head}.items():
        source = output / "sources" / name
        source.parent.mkdir(exist_ok=True)

        commit = checkout(ref, source)
        sources[name] = source
        (output / f"{name}-commit.txt").write_text(commit + "\n")

    (output / "environment.txt").write_text(
        f"{platform.platform()}\n" f"Logical CPUs: {os.cpu_count()}\n" f"Python: {sys.version}\n"
    )

    print("Preparing shared data...", flush=True)

    with (output / "data-preparation.log").open("w") as log:
        execute(
            [sys.executable, suite / "prepare_data.py", data],
            env={**env, "PYTHONPATH": str(REPO)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    with (output / "input.sha256").open("w") as f:
        for name in config["datasets"]:
            paths = [data / f"{name}.json", *sorted((data / f"{name}.zarr").rglob("*"))]
            for path in paths:
                if path.is_file():
                    with path.open("rb") as dataset:
                        digest = hashlib.file_digest(dataset, "sha256").hexdigest()
                    f.write(f"{digest}  {path.relative_to(data).as_posix()}\n")

    with (output / "dependencies.txt").open("w") as f:
        execute(
            ["uv", "pip", "freeze", "--python", sys.executable],
            stdout=f,
        )

    summaries = []

    for n_files in file_counts:
        case = output if len(file_counts) == 1 else output / f"{n_files}_files"
        case.mkdir(parents=True, exist_ok=True)

        if case != output:
            shutil.copy(output / "config.yaml", case)

        case_env = {**env, "BENCHMARK_N_FILES": str(n_files)}

        for name, source in sources.items():
            print(f"Benchmarking {name}: {n_files} RAW files...", flush=True)

            execute(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"--confcutdir={suite}",
                    "-o",
                    "python_files=benchmark_*.py",
                    "-o",
                    "python_functions=benchmark_*",
                    "-p",
                    "pytest_benchmark.plugin",
                    "--benchmark-only",
                    f"--benchmark-json={case / f'{name}.json'}",
                    "-q",
                    "--",
                    benchmark,
                ],
                cwd=suite,
                env={
                    **case_env,
                    "BENCHMARK_SOURCE": str(source),
                    "BENCHMARK_OUTPUT": str(case / name),
                    "PYTHONPATH": os.pathsep.join([str(source), str(suite)]),
                },
            )

        with (case / "summary.md").open("w") as f:
            execute(
                [sys.executable, suite / "report.py", case],
                env=case_env,
                stdout=f,
            )

        summaries.append(f"## {n_files} RAW files\n\n" + (case / "summary.md").read_text())

    if len(file_counts) > 1:
        (output / "summary.md").write_text("\n\n".join(summaries))

    print((output / "summary.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="main")
    parser.add_argument("--head", required=True)
    parser.add_argument("--benchmark", default=".")
    parser.add_argument("--output", default="benchmark-results")
    parser.add_argument("--config", type=Path, default=SUITE / "config.yaml")

    args = parser.parse_args()

    run(
        args.base,
        args.head,
        args.benchmark,
        args.output,
        args.config,
    )
