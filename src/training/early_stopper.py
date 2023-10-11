class EarlyStopper:
    def __init__(self, patience: int = 7, delta: float = 0.0):
        self._patience = patience
        self._delta = delta
        self.losses = []

    def stop(self):
        if len(self.losses) < self._patience:
            return False

        for i in range(1, self._patience):
            if self.losses[-i - 1] - self._delta > self.losses[-i]:
                return False

        return True
