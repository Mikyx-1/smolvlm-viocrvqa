"""Experiment tracking, wrapped so that --no-wandb needs no `if run:` checks."""


class RunTracker:
    """Log scalars to a wandb run, or to nothing at all."""

    def __init__(self, run=None):
        self.run = run

    def __bool__(self):
        """True when an actual run is attached."""
        return self.run is not None

    @classmethod
    def create(cls, enabled=True, project=None, name=None, config=None):
        """Start a wandb run, or return a no-op tracker when disabled."""
        if not enabled:
            return cls(None)
        import wandb

        return cls(wandb.init(project=project, name=name, config=config))

    def log(self, values, step=None):
        """Log a dict of scalars at `step`; a no-op when tracking is off."""
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self):
        """Close the run if there is one."""
        if self.run is not None:
            self.run.finish()
