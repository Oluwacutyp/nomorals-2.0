"""Error Intelligence System - Advanced error catching and root cause analysis.

This module provides intelligent error handling that:
1. Catches all exceptions with full context
2. Classifies errors by type and severity
3. Performs root cause analysis
4. Suggests fixes and workarounds
5. Maintains a knowledge base of common errors
6. Provides human-readable explanations

Usage:
    from nomorals.core.error_intelligence import ErrorIntelligence, catch_and_analyze
    
    ei = ErrorIntelligence()
    
    # Wrap any operation
    try:
        result = await some_risky_operation()
    except Exception as e:
        analysis = ei.analyze(e, context={"operation": "email_send", "account": "bot@gmail.com"})
        print(analysis.explanation)  # Human-readable
        print(analysis.suggested_fix)  # Actionable fix
        print(analysis.root_cause)  # What actually went wrong
    
    # Or use the decorator
    @catch_and_analyze(context={"operation": "calendar_sync"})
    async def sync_calendar():
        ...
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from ..core.logging_setup import get_logger
from .errors import NoMoralsError

__all__ = [
    "ErrorIntelligence",
    "ErrorAnalysis",
    "ErrorCategory",
    "ErrorSeverity",
    "catch_and_analyze",
    "ErrorKnowledgeBase",
]

_log = get_logger(__name__)


class ErrorCategory(Enum):
    """Classification of error types."""
    
    NETWORK = "network"  # Connection, timeout, DNS
    AUTH = "authentication"  # Invalid credentials, expired tokens, permissions
    RATE_LIMIT = "rate_limit"  # API rate limits, quota exceeded
    VALIDATION = "validation"  # Invalid input, schema errors
    RESOURCE = "resource"  # File not found, disk full, memory
    PARSING = "parsing"  # JSON/XML/HTML parsing failures
    INTEGRATION = "integration"  # Third-party API errors
    CRYPTO = "cryptography"  # Encryption/decryption failures
    DATABASE = "database"  # SQL errors, connection pool
    CONFIGURATION = "configuration"  # Missing config, invalid settings
    PERMISSION = "permission"  # File permissions, access denied
    TIMEOUT = "timeout"  # Operation exceeded time limit
    UNKNOWN = "unknown"  # Unclassified errors


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"  # Cosmetic, non-critical
    MEDIUM = "medium"  # Degraded functionality
    HIGH = "high"  # Feature broken, user impacted
    CRITICAL = "critical"  # System failure, data loss risk


@dataclass
class ErrorAnalysis:
    """Complete analysis of an error."""
    
    # Basic info
    exception_type: str
    exception_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    
    # Analysis
    root_cause: str
    explanation: str
    suggested_fix: str
    
    # Context
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    retryable: bool = False
    retry_after: Optional[float] = None  # Seconds to wait before retry
    error_code: str = ""
    
    # Related
    related_errors: list[str] = field(default_factory=list)
    documentation_url: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "context": self.context,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
            "documentation_url": self.documentation_url,
        }
    
    def to_user_message(self) -> str:
        """Generate user-friendly error message."""
        msg = f"⚠️ {self.explanation}\n\n"
        if self.suggested_fix:
            msg += f"💡 **Fix:** {self.suggested_fix}\n\n"
        if self.retryable:
            retry_text = "in a moment" if not self.retry_after else f"in {int(self.retry_after)} seconds"
            msg += f"🔄 This is temporary - I'll retry {retry_text}."
        return msg.strip()


class ErrorKnowledgeBase:
    """Database of known errors and their solutions."""
    
    # Pattern matchers for error messages
    PATTERNS = {
        # Network errors
        "connection_refused": {
            "patterns": [r"connection refused", r"connection reset", r"ECONNREFUSED"],
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Service is down or not accepting connections",
            "fix": "Check if the service is running and accessible. Verify the host/port configuration.",
        },
        "dns_failure": {
            "patterns": [r"name resolution failed", r"getaddrinfo failed", r"DNS.*not found"],
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "DNS lookup failed - domain doesn't exist or DNS server unreachable",
            "fix": "Check your internet connection and verify the domain name is correct.",
        },
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"deadline exceeded"],
            "category": ErrorCategory.TIMEOUT,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Operation took too long to complete",
            "fix": "The service is slow or overloaded. Try again later or increase the timeout value.",
            "retryable": True,
        },
        
        # Authentication errors
        "invalid_credentials": {
            "patterns": [r"invalid.*password", r"authentication failed", r"unauthorized", r"401"],
            "category": ErrorCategory.AUTH,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Wrong username or password",
            "fix": "Verify credentials are correct. If using OAuth, the token may have expired - refresh it.",
        },
        "expired_token": {
            "patterns": [r"token.*expired", r"invalid.*token", r"access.*denied"],
            "category": ErrorCategory.AUTH,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "OAuth token or API key has expired",
            "fix": "Refresh the OAuth token or generate a new API key. Check token expiry settings.",
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"forbidden", r"403", r"access denied"],
            "category": ErrorCategory.PERMISSION,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Insufficient permissions for this operation",
            "fix": "Check that the account has the required permissions/scopes. Re-authorize if needed.",
        },
        
        # Rate limiting
        "rate_limit": {
            "patterns": [r"rate.*limit", r"too many requests", r"429", r"quota.*exceeded"],
            "category": ErrorCategory.RATE_LIMIT,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "API rate limit exceeded",
            "fix": "Wait before making more requests. Consider using exponential backoff or reducing request frequency.",
            "retryable": True,
        },
        
        # Resource errors
        "file_not_found": {
            "patterns": [r"no such file", r"file not found", r"does not exist"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "File or directory doesn't exist",
            "fix": "Verify the file path is correct. Check if the file was moved or deleted.",
        },
        "disk_full": {
            "patterns": [r"no space left", r"disk.*full", r"out of space"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.CRITICAL,
            "root_cause": "Disk is full",
            "fix": "Free up disk space by deleting old files or increasing storage quota.",
        },
        "memory_error": {
            "patterns": [r"out of memory", r"memory allocation", r"killed.*oom"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.CRITICAL,
            "root_cause": "Out of memory",
            "fix": "Reduce memory usage or increase available RAM. Consider processing data in smaller chunks.",
        },
        
        # Parsing errors
        "json_parse": {
            "patterns": [r"json.*decode", r"invalid.*json", r"expecting.*value"],
            "category": ErrorCategory.PARSING,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Invalid JSON data",
            "fix": "The API returned malformed JSON. Check the response format or contact the service provider.",
        },
        "html_parse": {
            "patterns": [r"html.*parse", r"selector.*not found", r"element.*not found"],
            "category": ErrorCategory.PARSING,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "HTML structure changed or element not found",
            "fix": "The website may have updated their layout. Update the selectors or try a different method.",
        },
        
        # Integration errors
        "api_error": {
            "patterns": [r"api.*error", r"service.*unavailable", r"500.*internal"],
            "category": ErrorCategory.INTEGRATION,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Third-party API returned an error",
            "fix": "The external service is experiencing issues. Try again later or check their status page.",
            "retryable": True,
        },
        
        # Database errors
        "database_locked": {
            "patterns": [r"database.*locked", r"sqlite.*busy"],
            "category": ErrorCategory.DATABASE,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Database is locked by another process",
            "fix": "Another process is using the database. Wait and retry, or check for stuck transactions.",
            "retryable": True,
        },
        
        # Crypto errors
        "decryption_failed": {
            "patterns": [r"decryption.*failed", r"cipher.*error", r"invalid.*key"],
            "category": ErrorCategory.CRYPTO,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Decryption failed - wrong key or corrupted data",
            "fix": "Verify the encryption key is correct. The data may be corrupted or encrypted with a different key.",
        },
    }
    
    @classmethod
    def match(cls, error_message: str) -> Optional[dict[str, Any]]:
        """Match error message against known patterns.
        
        Args:
            error_message: Error message to match
            
        Returns:
            Matched pattern info or None
        """
        error_lower = error_message.lower()
        
        for pattern_name, pattern_info in cls.PATTERNS.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    return {
                        "pattern_name": pattern_name,
                        **pattern_info,
                    }
        
        return None


class ErrorIntelligence:
    """Intelligent error analysis and root cause detection."""
    
    def __init__(self) -> None:
        self.knowledge_base = ErrorKnowledgeBase()
        _log.info("Error Intelligence System initialized")
    
    def analyze(
        self,
        exception: Exception,
        *,
        context: dict[str, Any] | None = None,
        include_stack: bool = True,
    ) -> ErrorAnalysis:
        """Analyze an exception and provide insights.
        
        Args:
            exception: The exception to analyze
            context: Additional context about the operation
            include_stack: Include stack trace in analysis
            
        Returns:
            ErrorAnalysis with full details
        """
        context = context or {}
        
        # Get basic info
        exc_type = type(exception).__name__
        exc_message = str(exception)
        stack_trace = traceback.format_exc() if include_stack else ""
        
        # Try to match against knowledge base
        match = self.knowledge_base.match(exc_message)
        
        if match:
            # Known error pattern
            category = match["category"]
            severity = match["severity"]
            root_cause = match["root_cause"]
            explanation = self._generate_explanation(match, exception, context)
            suggested_fix = match["fix"]
            retryable = match.get("retryable", False)
            error_code = match["pattern_name"]
        else:
            # Unknown error - classify by exception type
            category = self._classify_by_type(exception)
            severity = self._estimate_severity(exception)
            root_cause = self._infer_root_cause(exception, context)
            explanation = self._generate_explanation_for_unknown(exception, context)
            suggested_fix = self._suggest_fix_for_unknown(exception, category)
            retryable = self._is_retryable(exception)
            error_code = f"unknown_{exc_type.lower()}"
        
        # Extract retry-after if available
        retry_after = self._extract_retry_after(exception)
        
        # Build analysis
        analysis = ErrorAnalysis(
            exception_type=exc_type,
            exception_message=exc_message,
            category=category,
            severity=severity,
            root_cause=root_cause,
            explanation=explanation,
            suggested_fix=suggested_fix,
            context=context,
            stack_trace=stack_trace,
            retryable=retryable,
            retry_after=retry_after,
            error_code=error_code,
        )
        
        # Log the analysis
        self._log_analysis(analysis)
        
        return analysis
    
    def _classify_by_type(self, exception: Exception) -> ErrorCategory:
        """Classify error by exception type."""
        exc_type = type(exception)
        exc_name = exc_type.__name__.lower()
        
        # Network errors
        if any(name in exc_name for name in ["connection", "socket", "http", "urllib"]):
            return ErrorCategory.NETWORK
        
        # Timeout errors
        if "timeout" in exc_name:
            return ErrorCategory.TIMEOUT
        
        # Permission errors
        if "permission" in exc_name or "access" in exc_name:
            return ErrorCategory.PERMISSION
        
        # File/resource errors
        if any(name in exc_name for name in ["file", "io", "os"]):
            return ErrorCategory.RESOURCE
        
        # Parsing errors
        if any(name in exc_name for name in ["json", "xml", "parse", "decode"]):
            return ErrorCategory.PARSING
        
        # Validation errors
        if any(name in exc_name for name in ["value", "type", "key", "attribute"]):
            return ErrorCategory.VALIDATION
        
        # Database errors
        if any(name in exc_name for name in ["sql", "database", "sqlite"]):
            return ErrorCategory.DATABASE
        
        # Auth errors
        if any(name in exc_name for name in ["auth", "credential", "token"]):
            return ErrorCategory.AUTH
        
        return ErrorCategory.UNKNOWN
    
    def _estimate_severity(self, exception: Exception) -> ErrorSeverity:
        """Estimate error severity."""
        exc_name = type(exception).__name__.lower()
        exc_message = str(exception).lower()
        
        # Critical errors
        if any(term in exc_message for term in ["out of memory", "disk full", "data loss"]):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if any(term in exc_name for term in ["auth", "permission", "crypto"]):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if any(term in exc_name for term in ["timeout", "rate", "parse"]):
            return ErrorSeverity.MEDIUM
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _infer_root_cause(self, exception: Exception, context: dict[str, Any]) -> str:
        """Infer root cause from exception and context."""
        exc_type = type(exception).__name__
        exc_message = str(exception)
        
        # Build root cause description
        cause = f"{exc_type}: {exc_message}"
        
        if context:
            operation = context.get("operation", "unknown operation")
            cause = f"Failed during {operation}. {cause}"
        
        return cause
    
    def _generate_explanation(
        self,
        match: dict[str, Any],
        exception: Exception,
        context: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation for known error."""
        operation = context.get("operation", "operation")
        service = context.get("service", "service")
        
        explanation = f"The {operation} failed because {match['root_cause'].lower()}"
        
        if service:
            explanation += f" while connecting to {service}"
        
        return explanation + "."
    
    def _generate_explanation_for_unknown(
        self,
        exception: Exception,
        context: dict[str, Any],
    ) -> str:
        """Generate explanation for unknown error."""
        operation = context.get("operation", "operation")
        exc_type = type(exception).__name__
        
        return f"An unexpected error occurred during {operation}: {exc_type}."
    
    def _suggest_fix_for_unknown(
        self,
        exception: Exception,
        category: ErrorCategory,
    ) -> str:
        """Suggest fix for unknown error."""
        if category == ErrorCategory.NETWORK:
            return "Check your internet connection and try again."
        elif category == ErrorCategory.AUTH:
            return "Verify your credentials are correct and not expired."
        elif category == ErrorCategory.RESOURCE:
            return "Check available system resources (disk space, memory)."
        elif category == ErrorCategory.TIMEOUT:
            return "The operation took too long. Try again or increase timeout."
        else:
            return "This is an unexpected error. Check the logs for details or try again."
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Determine if error is retryable."""
        exc_name = type(exception).__name__.lower()
        exc_message = str(exception).lower()
        
        # Retryable patterns
        retryable_patterns = [
            "timeout", "temporary", "unavailable", "busy", "locked",
            "rate limit", "429", "503", "504",
        ]
        
        if any(pattern in exc_name or pattern in exc_message for pattern in retryable_patterns):
            return True
        
        # Non-retryable patterns
        non_retryable_patterns = [
            "invalid", "unauthorized", "forbidden", "not found",
            "permission", "authentication",
        ]
        
        if any(pattern in exc_message for pattern in non_retryable_patterns):
            return False
        
        # Default: not retryable
        return False
    
    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Extract retry-after value from exception if available."""
        # Check if exception has retry_after attribute
        if hasattr(exception, "retry_after"):
            return getattr(exception, "retry_after")
        
        # Check if exception has response with Retry-After header
        if hasattr(exception, "response"):
            response = getattr(exception, "response")
            if hasattr(response, "headers"):
                headers = getattr(response, "headers")
                if "Retry-After" in headers:
                    try:
                        return float(headers["Retry-After"])
                    except (ValueError, TypeError):  # noqa: E103 - unparseable header treated as absent
                        pass
        
        return None
    
    def _log_analysis(self, analysis: ErrorAnalysis) -> None:
        """Log error analysis."""
        log_method = {
            ErrorSeverity.LOW: _log.info,
            ErrorSeverity.MEDIUM: _log.warning,
            ErrorSeverity.HIGH: _log.error,
            ErrorSeverity.CRITICAL: _log.critical,
        }.get(analysis.severity, _log.error)
        
        log_method(
            f"Error Analysis: [{analysis.category.value}] {analysis.exception_type}: "
            f"{analysis.explanation}"
        )
        
        if analysis.suggested_fix:
            _log.info(f"Suggested fix: {analysis.suggested_fix}")


# ── Decorator for easy error catching ──────────────────────────────────────────

T = TypeVar("T")


def catch_and_analyze(
    context: dict[str, Any] | None = None,
    *,
    reraise: bool = False,
    default: Any = None,
) -> Callable:
    """Decorator that catches exceptions and provides analysis.
    
    Args:
        context: Context to include in analysis
        reraise: If True, re-raise exception after analysis
        default: Default return value on error (if not reraising)
        
    Returns:
        Decorator function
        
    Example:
        @catch_and_analyze(context={"operation": "send_email"})
        async def send_email(to, subject, body):
            ...
    """
    ei = ErrorIntelligence()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Add function info to context
                full_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {}),
                }
                
                analysis = ei.analyze(e, context=full_context)
                
                # Store analysis on exception for later access
                e.error_analysis = analysis
                
                if reraise:
                    raise
                
                return default
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                full_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {}),
                }
                
                analysis = ei.analyze(e, context=full_context)
                e.error_analysis = analysis
                
                if reraise:
                    raise
                
                return default
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ── Global instance for convenience ────────────────────────────────────────────

_global_ei = ErrorIntelligence()


def analyze_error(
    exception: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> ErrorAnalysis:
    """Convenience function to analyze an error.
    
    Args:
        exception: Exception to analyze
        context: Additional context
        
    Returns:
        ErrorAnalysis object
    """
    return _global_ei.analyze(exception, context=context)
