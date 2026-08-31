"""Single-container entrypoint: migrations, then every worker and the gateway.

The compose stack runs these as five services. This runs them as five processes
in one container, because a demo platform deploy of five ~2GB images is five
build-and-boot cycles to get wrong before a deadline.

WHAT THIS GIVES UP, so it is a decision and not an accident:

  * The CPU-preprocess worker loses network isolation. In compose it sits on an
    `internal: true` network with no route off the host, read_only, cap_drop
    ALL, non-root -- and CLAUDE.md states plainly that this isolation IS the
    AV-scanning substitute, shipped alongside the worker because a parser of
    untrusted media without it is an open compromise window. Here it shares a
    container, a network namespace and a filesystem with the gateway. Acceptable
    for a time-boxed demo over known inputs. Not acceptable for public traffic;
    deploy the compose topology for that.
  * Per-service scaling. The GPU worker is the cost driver and the thing an HPA
    would scale independently; here it scales with everything else or not at all.
  * Process-level fault isolation. A worker that dies takes its restart policy
    from this launcher rather than from the platform.

Failing fast is deliberate throughout. A container that boots and then cannot
serve is worse on a platform than one that never reports healthy, because the
platform will route traffic to the first and not the second.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

# Order matters. Migrations must finish before anything opens a connection that
# assumes the schema, and the gateway must come last so it is not accepting
# uploads while the workers behind it are still starting.
WORKERS = [
    ("cpu-preprocess", [sys.executable, "-m", "df.workers.cpu_preprocess"]),
    ("gpu-inference", [sys.executable, "-m", "df.workers.gpu_inference"]),
    ("router", [sys.executable, "-m", "df.workers.router"]),
    ("retention-sweeper", [sys.executable, "-m", "df.workers.retention_sweeper"]),
]

_procs: list[tuple[str, subprocess.Popen]] = []


def _log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def _migrate() -> None:
    """Apply migrations before any worker connects.

    Synchronous and fatal on failure: a container serving against a schema it
    expects but does not have will fail on the first real request instead, which
    is far harder to read in a platform log than a refusal at boot.
    """
    _log("applying migrations")
    result = subprocess.run(
        [sys.executable, "scripts/migrate.py"], capture_output=True, text=True
    )
    for line in (result.stdout + result.stderr).splitlines():
        _log(f"  migrate: {line}")
    if result.returncode != 0:
        raise SystemExit("migrations failed; refusing to start")


def _shutdown(signum, _frame) -> None:
    _log(f"signal {signum}: stopping {len(_procs)} process(es)")
    for name, proc in _procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 15
    for name, proc in _procs:
        try:
            proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            _log(f"  {name} did not stop; killing")
            proc.kill()
    raise SystemExit(0)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _migrate()

    for name, cmd in WORKERS:
        _log(f"starting {name}")
        _procs.append((name, subprocess.Popen(cmd)))

    # The gateway binds the port the platform health-checks, so it goes last.
    # PORT is injected by most platforms and is not necessarily 8000.
    port = os.environ.get("PORT", "8000")
    _log(f"starting gateway on :{port}")
    _procs.append((
        "gateway",
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", "df.gateway.app:app",
            "--host", "0.0.0.0", "--port", port,
        ]),
    ))

    # Supervise. If any process exits, bring the whole container down rather
    # than limping: a container that is healthy on the port but has no inference
    # worker behind it accepts uploads and never answers them, which looks like
    # a hang to the user and like success to the platform.
    while True:
        for name, proc in _procs:
            code = proc.poll()
            if code is not None:
                _log(f"{name} exited with {code}; shutting down the container")
                for other_name, other in _procs:
                    if other is not proc and other.poll() is None:
                        other.terminate()
                return code or 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
