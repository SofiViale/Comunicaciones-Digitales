class SynchronizationError(Exception):
    """Base class for synchronization-related errors."""
    pass

class NoCandidatesFoundError(SynchronizationError):
    """No synchronization candidates were found."""
    pass

class SFDError(SynchronizationError):
    """Failed to locate the downchirp pair in SFD."""
    pass

class IncompleteFrameError(SynchronizationError):
    """IQ buffer too short to extract the body."""
    pass

class IncompleteHeaderError(IncompleteFrameError):
    """IQ buffer too short to extract the header."""
    pass

class IncompletePayloadError(IncompleteFrameError):
    """IQ buffer too short to extract as many payload symbols as the header indicates."""
    pass

class CandidatesExhaustedError(SynchronizationError):
    """All candidates were exhausted without successful synchronization."""
    pass
