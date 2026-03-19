import time
from threading import Lock


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        success_threshold: int = 2,
    ):
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._lock = Lock()

    def call_allowed(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    return True
                return False
            return True

    def record_success(self):
        with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == "CLOSED":
                self.failure_count = 0

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.success_count = 0
                return
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


deepface_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    success_threshold=2,
)
