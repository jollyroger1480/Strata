# Terminal quick start (source checkout)

These commands describe the current `Strata-TUI-CLI` development branch. Run them from the repository root with the Python environment used for Strata. Substitute `python3` on systems without a `python` command. They do not install a model or change the existing one-click setup path. The complete options, permissions and limits are in [TERMINAL.md](TERMINAL.md).

## Read a running server

With Strata already serving locally, get one machine-readable snapshot:

```sh
python strata.py status --url http://127.0.0.1:8080 --json
```

The interactive monitor is optional. Install its pinned dependency in that same Python environment, then open the one-screen Overview:

```sh
python -m pip install -r requirements-monitor.txt
python strata.py monitor --url http://127.0.0.1:8080
```

The Overview puts serving state, prompt and answer speed, batch/queue pressure, four selected GPU rows, requests and activity on one screen. Press `1`–`6` for detail, `Space` to freeze the display, `r` to refresh, and `q` to detach. Detach leaves the server running. On a machine without a terminal, use `status --json` or `monitor --once --json`.

For a remote server, use HTTPS or a local SSH tunnel. A saved connection stores only the credential environment variable's **name**:

```sh
python strata.py connection create work --url https://example.invalid --credential-env STRATA_TELEMETRY_KEY
python strata.py monitor --connection work
```

Set `STRATA_TELEMETRY_KEY` in the environment before connecting. Replace `example.invalid` with your own server root. The monitor client never needs a management credential.

## Prepare and rerun a local model

Use the run config written by setup. Start by reviewing its components and effective launch; neither command starts an engine:

```sh
python strata.py doctor --profile /absolute/path/to/strata-model.json --components --output json
python strata.py serve --profile /absolute/path/to/strata-model.json --plan --output json
```

For repeatable local management on POSIX, create a named profile from that config. Review changes before saving them:

```sh
python strata.py profile create four-gpu --from-config /absolute/path/to/strata-model.json
python strata.py profile set four-gpu parallel=8 batch_groups=auto max_queue=32 queue_timeout=15 batch_telemetry=true --plan
python strata.py profile set four-gpu parallel=8 batch_groups=auto max_queue=32 queue_timeout=15 batch_telemetry=true
python strata.py profile show four-gpu --effective --output json
python strata.py runtime start --profile four-gpu --plan --output json
python strata.py runtime start --profile four-gpu
python strata.py runtime verify --profile four-gpu --output json
```

Choose slot and queue settings for the actual model and card memory. A requested `parallel=8` does not prove the engine loaded eight effective slots. Check the reported values after start. To change a running profile, save the edit, review the restart, then execute it:

```sh
python strata.py profile set four-gpu parallel=4 --plan
python strata.py profile set four-gpu parallel=4
python strata.py runtime restart --profile four-gpu --plan --output json
python strata.py runtime restart --profile four-gpu
```

The restart is an explicit operation. `profile set` alone does not restart or retune a running model. `runtime stop --profile four-gpu` stops only the owned instance; `q` in Monitor does not. The POSIX runtime host and rollback path have fake-engine tests, but real GPU and Windows ownership qualification remains open.

## Check cards, calibrate and benchmark

Read installed components without launching the engine. The `--live` forms also probe host GPUs:

```sh
python strata.py components verify --profile four-gpu --output json
python strata.py doctor --profile four-gpu --live --output json
```

Calibration is separate from serving. Review the plan and measure a decode sweep only when the selected devices are available for exclusive use. Measuring writes a report but does not apply settings:

```sh
python strata.py engine calibrate --profile four-gpu --plan --output json
python strata.py engine calibrate --profile four-gpu --measure --exclusive --out /absolute/path/to/calibration.json
python strata.py engine calibrate --profile four-gpu --review-result /absolute/path/to/calibration.json --output json
```

After reviewing the report, applying and rolling back remain separate explicit actions. Both require exclusive device access:

```sh
python strata.py engine calibrate --profile four-gpu --apply-result /absolute/path/to/calibration.json --exclusive
python strata.py engine calibrate --profile four-gpu --review-rollback /absolute/path/to/calibration.json --output json
python strata.py engine calibrate --profile four-gpu --rollback-result /absolute/path/to/calibration.json --exclusive
```

Check the candidate report and owner state before either action. The full preconditions and rollback behavior are in [TERMINAL.md](TERMINAL.md).

After the model is loaded, save an explicit serving benchmark report:

```sh
python strata.py bench run --profile four-gpu --suite serving-smoke --concurrency 4 --out /absolute/path/to/new-bench-report
```

The benchmark writes into a new output directory and does not publish its result. Synthetic screenshots and local fake-engine tests do not establish performance on a four-card host. Capture the exact GPU generation, model, driver and workload before comparing real runs.
