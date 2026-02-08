"""Google OAuth2 token manager for Google Calendar integration.

Handles the full OAuth2 lifecycle:
- Authorization URL generation
- Authorization code → token exchange
- Token persistence (load / save to disk)
- Automatic access token refresh via refresh_token
- Backward compatibility with static AGENTARMY_GOOGLE_CALENDAR_TOKEN

Token storage: ~/.agentarmy/google_tokens.json (or project-local .agentarmy/)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Google OAuth2 endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Default scopes for Calendar API
DEFAULT_SCOPES = "https://www.googleapis.com/auth/calendar"


class GoogleOAuthTokenManager:
    """Manages Google OAuth2 tokens — storage, refresh, and validation.

    Supports two authentication modes:
    1. Full OAuth2 flow (client_id + client_secret + user consent)
    2. Static Bearer token fallback (AGENTARMY_GOOGLE_CALENDAR_TOKEN env var)

    Token file format (.agentarmy/google_tokens.json):
    {
        "access_token": "ya29.a0...",
        "refresh_token": "1//0e...",
        "expires_at": 1700000000.0,
        "token_type": "Bearer",
        "scope": "https://www.googleapis.com/auth/calendar",
        "email": "user@gmail.com"
    }
    """

    # Refresh 5 minutes before actual expiry
    EXPIRY_BUFFER_SECONDS = 300

    def __init__(self, token_path: Optional[Path] = None) -> None:
        """Initialize the token manager.

        Args:
            token_path: Path to the token JSON file. Defaults to
                        .agentarmy/google_tokens.json in the working directory
                        (or home directory as fallback).
        """
        if token_path:
            self.token_path = token_path
        else:
            # Prefer project-local .agentarmy/, fall back to home dir
            local = Path(".agentarmy") / "google_tokens.json"
            home = Path.home() / ".agentarmy" / "google_tokens.json"
            self.token_path = local if local.parent.exists() else home

        self._tokens: Optional[dict[str, Any]] = None
        logger.debug("GoogleOAuthTokenManager initialized", token_path=str(self.token_path))

    # ── Token persistence ─────────────────────────────────────────

    def load_tokens(self) -> Optional[dict[str, Any]]:
        """Load tokens from disk.

        Returns:
            Token dict if file exists and is valid JSON, else None.
        """
        if self._tokens:
            return self._tokens

        if not self.token_path.exists():
            logger.debug("No token file found", path=str(self.token_path))
            return None

        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
            self._tokens = data
            logger.info("Loaded Google OAuth tokens", email=data.get("email"))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load Google tokens", error=str(exc))
            return None

    def save_tokens(self, tokens: dict[str, Any]) -> None:
        """Save tokens to disk.

        Creates the parent directory if it doesn't exist.
        Sets file permissions to user-only (0600) on Unix.

        Args:
            tokens: Token dict to persist.
        """
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(
            json.dumps(tokens, indent=2),
            encoding="utf-8",
        )
        # Restrict permissions (Unix only)
        try:
            self.token_path.chmod(0o600)
        except OSError:
            pass

        self._tokens = tokens
        logger.info("Saved Google OAuth tokens", email=tokens.get("email"))

    def delete_tokens(self) -> None:
        """Delete the token file from disk."""
        if self.token_path.exists():
            self.token_path.unlink()
        self._tokens = None
        logger.info("Deleted Google OAuth tokens")

    # ── Token validation ──────────────────────────────────────────

    def is_token_valid(self, buffer_seconds: int = EXPIRY_BUFFER_SECONDS) -> bool:
        """Check if the current access token is still valid.

        Args:
            buffer_seconds: Seconds before expiry to consider token invalid.

        Returns:
            True if token exists and won't expire within buffer_seconds.
        """
        tokens = self.load_tokens()
        if not tokens or "access_token" not in tokens:
            return False

        expires_at = tokens.get("expires_at", 0)
        return time.time() < (expires_at - buffer_seconds)

    @property
    def has_tokens(self) -> bool:
        """Check if any tokens (OAuth or static) are available."""
        tokens = self.load_tokens()
        if tokens and tokens.get("access_token"):
            return True
        # Fallback to static token
        return bool(os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", ""))

    # ── Authorization URL ─────────────────────────────────────────

    @staticmethod
    def build_authorize_url(
        client_id: str,
        redirect_uri: str,
        scopes: str = DEFAULT_SCOPES,
        state: str = "",
    ) -> str:
        """Build the Google OAuth2 consent screen URL.

        Args:
            client_id: Google OAuth client ID.
            redirect_uri: Callback URL registered in Google Cloud Console.
            scopes: Space-separated OAuth scopes.
            state: CSRF protection token.

        Returns:
            Full authorization URL to redirect the user to.
        """
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scopes,
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    # ── Token exchange ────────────────────────────────────────────

    async def exchange_code(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Exchange an authorization code for access + refresh tokens.

        Args:
            code: Authorization code from the callback.
            client_id: Google OAuth client ID.
            client_secret: Google OAuth client secret.
            redirect_uri: Must match the one used in the authorization URL.

        Returns:
            Token dict with access_token, refresh_token, expires_at, etc.

        Raises:
            httpx.HTTPStatusError: If Google rejects the exchange.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        tokens: dict[str, Any] = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "token_type": data.get("token_type", "Bearer"),
            "scope": data.get("scope", ""),
            "expires_at": time.time() + data.get("expires_in", 3600),
        }

        # Fetch user email for display
        try:
            email = await self._fetch_user_email(tokens["access_token"])
            tokens["email"] = email
        except Exception:
            tokens["email"] = None

        self.save_tokens(tokens)
        logger.info("Exchanged authorization code for tokens", email=tokens.get("email"))
        return tokens

    # ── Token refresh ─────────────────────────────────────────────

    async def refresh_access_token(
        self,
        client_id: str,
        client_secret: str,
    ) -> str:
        """Refresh the access token using the stored refresh_token.

        Args:
            client_id: Google OAuth client ID.
            client_secret: Google OAuth client secret.

        Returns:
            New access token string.

        Raises:
            ValueError: If no refresh token is available.
            httpx.HTTPStatusError: If Google rejects the refresh.
        """
        tokens = self.load_tokens()
        if not tokens or not tokens.get("refresh_token"):
            raise ValueError("No refresh token available — user must re-authorize")

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": tokens["refresh_token"],
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Update stored tokens (refresh_token is NOT returned on refresh)
        tokens["access_token"] = data["access_token"]
        tokens["expires_at"] = time.time() + data.get("expires_in", 3600)
        if "scope" in data:
            tokens["scope"] = data["scope"]

        self.save_tokens(tokens)
        logger.info("Refreshed Google access token", email=tokens.get("email"))
        return tokens["access_token"]

    # ── Convenience: get a valid token ────────────────────────────

    async def get_valid_token(
        self,
        client_id: str = "",
        client_secret: str = "",
    ) -> str:
        """Get a valid access token, refreshing if needed.

        Falls back to the static AGENTARMY_GOOGLE_CALENDAR_TOKEN env var
        if no OAuth tokens exist.

        Args:
            client_id: Google OAuth client ID (needed for refresh).
            client_secret: Google OAuth client secret (needed for refresh).

        Returns:
            A valid access token string.

        Raises:
            ValueError: If no tokens are available at all.
        """
        # 1. Try OAuth tokens
        tokens = self.load_tokens()
        if tokens and tokens.get("access_token"):
            if self.is_token_valid():
                return tokens["access_token"]

            # Try refresh
            if tokens.get("refresh_token") and client_id and client_secret:
                try:
                    return await self.refresh_access_token(client_id, client_secret)
                except Exception as exc:
                    logger.warning("Token refresh failed", error=str(exc))

        # 2. Fallback to static token
        static_token = os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", "")
        if static_token:
            logger.debug("Using static Google Calendar token")
            return static_token

        raise ValueError(
            "Google Calendar not configured. "
            "Connect via Settings → Google Calendar, or set AGENTARMY_GOOGLE_CALENDAR_TOKEN."
        )

    # ── Token revocation ──────────────────────────────────────────

    async def revoke(self) -> bool:
        """Revoke the current tokens at Google and delete local file.

        Returns:
            True if revocation succeeded or tokens didn't exist.
        """
        tokens = self.load_tokens()
        if not tokens:
            return True

        token_to_revoke = tokens.get("access_token") or tokens.get("refresh_token", "")
        if token_to_revoke:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        GOOGLE_REVOKE_URL,
                        params={"token": token_to_revoke},
                    )
                    if resp.status_code not in (200, 400):
                        logger.warning("Google revoke returned unexpected status", status=resp.status_code)
            except Exception as exc:
                logger.warning("Failed to revoke token at Google", error=str(exc))

        self.delete_tokens()
        return True

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get the current connection status.

        Returns:
            Dict with connected, email, expires_at, using_static_token fields.
        """
        tokens = self.load_tokens()
        if tokens and tokens.get("access_token"):
            return {
                "connected": True,
                "email": tokens.get("email"),
                "expires_at": tokens.get("expires_at"),
                "using_static_token": False,
            }

        static_token = os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", "")
        if static_token:
            return {
                "connected": True,
                "email": None,
                "expires_at": None,
                "using_static_token": True,
            }

        return {
            "connected": False,
            "email": None,
            "expires_at": None,
            "using_static_token": False,
        }

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    async def _fetch_user_email(access_token: str) -> Optional[str]:
        """Fetch the user's email from Google userinfo endpoint.

        Args:
            access_token: Valid Google access token.

        Returns:
            Email string or None.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if resp.status_code == 200:
                    return resp.json().get("email")
        except Exception:
            pass
        return None
