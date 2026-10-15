"""Emergency parent-loss guard for a server launched by the local runtime host.

The inherited pipe has exactly one writer, held by the host.  EOF therefore
means the host exited, including after SIGKILL.  The server has its own process
group so its engine subprocesses receive the emergency stop as well.
"""

from __future__ import annotations

import os
import signal
import stat
import threading


def start(fd: int) -> None:
    if os.name != "posix" or fd < 0 or os.getpgrp() != os.getpid():
        raise ValueError("--owner-fd requires a dedicated POSIX server process group")
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise ValueError("--owner-fd must be an inherited pipe")

    def watch() -> None:
        try:
            while os.read(fd, 1):
                pass
            # This is an emergency path.  Signal the whole dedicated group so
            # an engine already spawned during model loading cannot be orphaned.
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except OSError:
            # A broken guard must never authorize another process to manage us.
            os.killpg(os.getpgrp(), signal.SIGTERM)
        finally:
            os.close(fd)

    threading.Thread(target=watch, name="strata-owner-guard", daemon=True).start()
