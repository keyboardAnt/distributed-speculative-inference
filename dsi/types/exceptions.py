class NumOfTargetServersInsufficientError(Exception):
    def __init__(
        self,
        message="The current analysis supports only simples cases where there are no waits on target servers. For every k drafts that are ready for verification, there must be an idle target server.",
    ):
        super().__init__(message)
