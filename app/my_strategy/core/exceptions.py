"""Project-specific exceptions."""


class KHQuantError(Exception):
    """Base project exception."""


class ValidationError(KHQuantError):
    """Raised when validation blocks a run."""


class DataSourceError(KHQuantError):
    """Raised when all configured data sources fail."""
