import contextvars
from typing import Optional

_buyer_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "buyer_id", default=None
)

def set_buyer_id(buyer_id: str) -> contextvars.Token:
    """
    Set the current buyer ID in the context and return a token
    that can be used to reset it if needed.
    """
    return _buyer_id.set(buyer_id)

def get_buyer_id() -> Optional[str]:
    """
    Get the current buyer ID from the context.
    Raises an error if the context is not set, or returns None if not required.
    For this application, we assume the buyer must be set for tools.
    """
    buyer_id = _buyer_id.get()
    if buyer_id is None:
        raise ValueError("buyer_id is not set in the current context.")
    return buyer_id
