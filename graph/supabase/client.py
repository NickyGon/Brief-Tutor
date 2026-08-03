"""
Supabase client factory with lazy initialization.
"""

from typing import Dict

from graph.supabase.config import SupabaseSettings

try:
    from supabase import Client, create_client
    from supabase.lib.client_options import ClientOptions
except Exception:  # pragma: no cover - handled at runtime with clear error
    Client = object
    create_client = None
    ClientOptions = None


_CLIENT_CACHE: Dict[str, "Client"] = {}


def get_supabase_client(use_service_role: bool = False) -> "Client":
    """
    Create or return a cached Supabase client.

    Args:
        use_service_role: If True, use SUPABASE_SERVICE_ROLE_KEY.
                          Otherwise uses SUPABASE_ANON_KEY.
    """
    if create_client is None:
        raise RuntimeError(
            "supabase package is not installed. Add 'supabase' to requirements "
            "and run pip install -r requirements.txt."
        )

    settings = SupabaseSettings.from_env()
    mode = "service_role" if use_service_role else "anon"

    if mode in _CLIENT_CACHE:
        return _CLIENT_CACHE[mode]

    if not settings.enabled:
        raise RuntimeError(
            "Supabase integration is disabled. Set SUPABASE_ENABLED=true in .env."
        )

    if not settings.url:
        raise RuntimeError("Missing SUPABASE_URL in environment.")

    key = settings.service_role_key if use_service_role else settings.anon_key
    key_name = "SUPABASE_SERVICE_ROLE_KEY" if use_service_role else "SUPABASE_ANON_KEY"
    if not key:
        raise RuntimeError(f"Missing {key_name} in environment.")

    options = None
    if ClientOptions is not None:
        try:
            # Prefer schema-aware options when the installed supabase SDK supports them.
            options = ClientOptions(schema=settings.schema)
        except Exception:
            options = None

    try:
        if options is not None:
            client = create_client(settings.url, key, options=options)
        else:
            client = create_client(settings.url, key)
    except AttributeError:
        # Some supabase SDK versions have ClientOptions/client mismatches.
        # Fall back to the default client (public schema).
        client = create_client(settings.url, key)
    _CLIENT_CACHE[mode] = client
    return client


def reset_supabase_clients() -> None:
    """
    Clear cached clients (useful for tests or env reloads).
    """
    _CLIENT_CACHE.clear()
