"""
业务异常定义模块
定义平台各环节的领域异常，便于上层统一捕获并映射为 HTTP 响应。
"""

class MIHCError(Exception):
    """平台基础异常。"""
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
    """输入被安全护栏拦截（敏感信息 / 注入攻击）。"""
    code = "input_guarded"
    status_code = 422


class OutputGuardError(MIHCError):
    """输出未通过安全审查。"""
    code = "output_guarded"
    status_code = 500


class RetrievalError(MIHCError):
    """检索链路异常（向量库/关键词库不可用等）。"""
    code = "retrieval_error"


class AgentError(MIHCError):
    """智能体执行失败。"""
    code = "agent_error"


class IntentError(MIHCError):
    """意图识别失败。"""
    code = "intent_error"


class ModelUnavailableError(MIHCError):
    """模型服务不可用（所有候选模型均超时/失败）。"""
    code = "model_unavailable"
    status_code = 503


class PermissionDeniedError(MIHCError):
    """无权限访问指定资源。"""
    code = "permission_denied"
    status_code = 403


class SessionNotFoundError(MIHCError):
    """会话不存在。"""
    code = "session_not_found"
    status_code = 404
