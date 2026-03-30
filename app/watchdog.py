import threading
import time
import os
import requests


class Watchdog:
    """Auto-stops the RunPod pod after a period of inactivity."""

    def __init__(self, timeout_minutes: int = 15):
        self.timeout = timeout_minutes * 60
        self.last_activity = time.time()
        self._running = False

    def ping(self) -> None:
        """Call on every inference request to reset the idle timer."""
        self.last_activity = time.time()

    def start(self) -> None:
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            time.sleep(60)
            if time.time() - self.last_activity > self.timeout:
                self._stop_pod()

    def _stop_pod(self) -> None:
        pod_id = os.getenv("RUNPOD_POD_ID", "")
        api_key = os.getenv("RUNPOD_API_KEY", "")
        if pod_id and api_key:
            try:
                requests.post(
                    f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
            except Exception:
                pass


watchdog = Watchdog(timeout_minutes=15)
