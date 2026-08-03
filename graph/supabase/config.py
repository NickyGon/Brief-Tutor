"""
Configuration utilities for Supabase connectivity.
"""

from dataclasses import dataclass
import os


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str
    service_role_key: str
    schema: str
    enabled: bool

    @classmethod
    def from_env(cls) -> "SupabaseSettings":
        return cls(
            url=str(os.getenv("SUPABASE_URL", "")).strip(),
            anon_key=str(os.getenv("SUPABASE_ANON_KEY", "")).strip(),
            service_role_key=str(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip(),
            schema=str(os.getenv("SUPABASE_SCHEMA", "public")).strip() or "public",
            enabled=_truthy(str(os.getenv("SUPABASE_ENABLED", "false"))),
        )

    def is_configured(self, use_service_role: bool = False) -> bool:
        if not self.enabled or not self.url:
            return False
        if use_service_role:
            return bool(self.service_role_key)
        return bool(self.anon_key)


def is_supabase_configured(use_service_role: bool = False) -> bool:
    settings = SupabaseSettings.from_env()
    return settings.is_configured(use_service_role=use_service_role)
