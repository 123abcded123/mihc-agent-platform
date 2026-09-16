
class MIHCError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}

class InputGuardError(MIHCError):
    code = "input_guarded"
    status_code = 422

class OutputGuardError(MIHCError):
    code = "output_guarded"
    status_code = 500

class RetrievalError(MIHCError):
    code = "retrieval_error"

class AgentError(MIHCError):
    code = "agent_error"

class IntentError(MIHCError):
    code = "intent_error"

class ModelUnavailableError(MIHCError):
    code = "model_unavailable"
    status_code = 503

class PermissionDeniedError(MIHCError):
    code = "permission_denied"
    status_code = 403

class SessionNotFoundError(MIHCError):
    code = "session_not_found"
    status_code = 404
