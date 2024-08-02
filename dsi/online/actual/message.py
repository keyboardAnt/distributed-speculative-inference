from dataclasses import dataclass


@dataclass
class MsgVerifiedRightmost:
    """
    The index and id of the last verified token.
    """

    v: int  # The index of the last verified token
    tok_id: int  # The last verified token id
    state: dict[str, int]  # The serialized state
