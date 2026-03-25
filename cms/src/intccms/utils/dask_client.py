"""Dask distributed client utilities for analysis facility environments.

Provides a unified interface for acquiring Dask clients across different
analysis facilities (CoffeaCasa Condor, CoffeaCasa Gateway, Purdue AF),
along with worker logging utilities.
"""

import logging
import os
import sys
import threading
from contextlib import contextmanager
import cloudpickle
from pathlib import Path
from typing import List, Optional, Tuple

from dask.distributed import Client, PipInstall, WorkerPlugin

logger = logging.getLogger(__name__)

# Register modules for cloud pickle (not sure why we need to do this)
cloudpickle.register_pickle_by_value(sys.modules[__name__])

class PrintForwarder(WorkerPlugin):
    """Dask WorkerPlugin that redirects worker stdout/stderr to the event log.

    Intercepts print statements on workers and forwards them as dask events
    under the ``"prints"`` topic, enabling client-side polling via
    :func:`live_prints`.
    """

    def setup(self, worker):
        import sys

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        addr = worker.address

        class _Stream:
            def __init__(self, name, orig):
                self.name = name
                self.orig = orig

            def write(self, msg):
                if msg.strip():
                    try:
                        worker.log_event(
                            "prints",
                            f"[{addr}] [{self.name}] {msg.rstrip()}",
                        )
                    except Exception:
                        pass
                    self.orig.write(msg)
                return len(msg) if msg else 0

            def flush(self):
                self.orig.flush()

        sys.stdout = _Stream("out", orig_stdout)
        sys.stderr = _Stream("err", orig_stderr)


class _AWSEnvPlugin(WorkerPlugin):
    """WorkerPlugin that sets AWS credentials on each worker.

    Captures credentials at plugin creation time (on the client) and
    injects them into ``os.environ`` on every worker during setup.
    """

    def __init__(self, key: str, secret: str):
        self._key = key
        self._secret = secret

    def setup(self, worker):
        import os as _os

        if self._key:
            _os.environ["AWS_ACCESS_KEY_ID"] = self._key
        if self._secret:
            _os.environ["AWS_SECRET_ACCESS_KEY"] = self._secret


def live_prints(client: Client, interval: float = 1.0) -> threading.Event:
    """Poll worker print events and display them on the client side.

    Starts a daemon thread that periodically checks for new ``"prints"``
    events logged by :class:`PrintForwarder` on workers.

    Args:
        client: Active Dask client with PrintForwarder registered.
        interval: Polling interval in seconds.

    Returns:
        threading.Event: Set this event to stop the polling thread.
    """
    seen = 0
    stop = threading.Event()

    def _poll():
        nonlocal seen
        while not stop.is_set():
            events = client.get_events("prints")
            for _ts, msg in events[seen:]:
                print(msg)
            seen = len(events)
            stop.wait(interval)

    threading.Thread(target=_poll, daemon=True).start()
    return stop


@contextmanager
def acquire_client(
    af: str,
    num_workers: Optional[int] = None,
    close_after: bool = False,
    pip_packages: Optional[List[str]] = None,
    propagate_aws_env: bool = False,
) -> Tuple[Client, Optional[object]]:
    """Context manager to acquire a Dask client for a given analysis facility.

    Supported analysis facilities:

    - ``"coffeacasa-condor"``: Direct connection to ``tls://localhost:8786``.
    - ``"coffeacasa-gateway"``: Connects via ``dask_gateway.Gateway()`` with
      X509 proxy setup and access token upload.
    - ``"purdue-af"``: Connects via ``dask_gateway.Gateway()`` with minimal
      setup.

    :class:`PrintForwarder` is always registered. ``client.forward_logging()``
    is always called. ``PipInstall`` is registered only when *pip_packages* is
    provided.

    Args:
        af: Analysis facility identifier.
        num_workers: Number of workers to request (gateway AFs only).
            None means no scaling is performed.
        close_after: If True, close the client when exiting the context.
        pip_packages: Optional list of pip package specifiers to install
            on workers via PipInstall (e.g., ``["coffea==2025.12.0"]``).
        propagate_aws_env: If True, capture ``AWS_ACCESS_KEY_ID`` and
            ``AWS_SECRET_ACCESS_KEY`` from the client environment and set
            them on all workers.

    Yields:
        Tuple of ``(client, cluster)`` where *cluster* is ``None`` for
        coffeacasa-condor.
    """
    client, cluster = None, None

    try:
        if af == "coffeacasa-condor":
            client = Client("tls://localhost:8786")

        elif af == "coffeacasa-gateway":
            from dask_gateway import Gateway

            def _set_gateway_env(dask_worker):
                config_path = str(
                    Path(dask_worker.local_directory) / "access_token"
                )
                os.environ["BEARER_TOKEN_FILE"] = config_path
                os.chmod(config_path, 0o600)
                os.chmod("/etc/grid-security/certificates", 0o755)
                config_path2 = str(
                    Path(dask_worker.local_directory) / "x509up_u6440"
                )
                os.environ["X509_USER_PROXY"] = config_path2
                os.chmod(config_path2, 0o600)

            gateway = Gateway()
            clusters = gateway.list_clusters()
            cluster = gateway.connect(clusters[0].name)
            client = cluster.get_client()
            if num_workers is not None:
                cluster.scale(num_workers)
                client.wait_for_workers(num_workers)
            client.upload_file("/etc/cmsaf-secrets-chown/access_token")
            client.upload_file("/tmp/x509up_u6440")
            client.register_worker_callbacks(setup=_set_gateway_env)

        elif af == "purdue-af":
            from dask_gateway import Gateway

            gateway = Gateway()
            clusters = gateway.list_clusters()
            cluster = gateway.connect(clusters[0].name)
            client = cluster.get_client()
            if num_workers is not None:
                cluster.scale(num_workers)
                client.wait_for_workers(num_workers)

        else:
            raise NotImplementedError(
                f"Analysis facility '{af}' is not supported. "
                f"Supported: coffeacasa-condor, coffeacasa-gateway, purdue-af"
            )

        # Propagate AWS credentials to workers
        if propagate_aws_env:
            client.register_plugin(
                _AWSEnvPlugin(
                    key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                    secret=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                )
            )

        # Register PrintForwarder on all AFs
        client.register_plugin(PrintForwarder())

        # Install pip packages if requested
        if pip_packages:
            _is_gateway = af in ("coffeacasa-gateway", "purdue-af")
            _has_git = any("git" in pkg for pkg in pip_packages)
            if _is_gateway and _has_git:
                raise ValueError(
                    "Gateway facilities do not support installing packages "
                    "from git URLs on workers. Use PyPI packages instead."
                )
            client.register_plugin(PipInstall(packages=pip_packages))

        # Forward logging
        try:
            client.forward_logging()
        except Exception:
            pass

        logger.info("Connected to Dask scheduler")
        logger.info("Dashboard: %s", client.dashboard_link)

        yield client, cluster

    finally:
        if close_after and client is not None:
            client.close()
            logger.info("Client closed")
