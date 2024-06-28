class NumOfTargetServersInsufficientError(Exception):
    def __init__(
        self,
        msg: str,
    ):
        prefix: str = (
            "The current analysis supports only simples cases where there are no waits on target servers. For every k drafts that are ready for verification, there must be an idle target server. "
        )
        super().__init__(prefix + msg)


class DrafterSlowerThanTargetError(Exception):
    def __init__(
        self,
        msg: str,
    ):
        prefix: str = (
            "The drafter must be as fast as the target or faster than the target. "
        )
        super().__init__(prefix + msg)
