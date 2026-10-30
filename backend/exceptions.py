from __future__ import annotations


class AppException(Exception):
    def __init__(self, status_code: int, error_code: str, message: str):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class BadRequestException(AppException):
    def __init__(self, message: str, error_code: str = "BAD_REQUEST"):
        super().__init__(status_code=400, error_code=error_code, message=message)


class UnauthorizedException(AppException):
    def __init__(self, message: str = "Authentication required", error_code: str = "UNAUTHENTICATED"):
        super().__init__(status_code=401, error_code=error_code, message=message)


class ForbiddenException(AppException):
    def __init__(self, message: str = "Access forbidden", error_code: str = "UNAUTHORIZED"):
        super().__init__(status_code=403, error_code=error_code, message=message)


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found", error_code: str = "NOT_FOUND"):
        super().__init__(status_code=404, error_code=error_code, message=message)


class ConflictException(AppException):
    def __init__(self, message: str = "Resource conflict", error_code: str = "CONFLICT"):
        super().__init__(status_code=409, error_code=error_code, message=message)
