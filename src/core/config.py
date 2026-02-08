"""Configuration management for AgentArmy using Pydantic Settings."""

import logging
import secrets
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """System-level configuration settings.

    Attributes:
        system_name: Name of the agent system.
        environment: Deployment environment (development, staging, production).
        debug: Enable debug mode.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    system_name: str = Field(default="AgentArmy", description="System name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_SYSTEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class RedisSettings(BaseSettings):
    """Redis configuration settings.

    Attributes:
        host: Redis server host.
        port: Redis server port.
        db: Redis database number.
        password: Redis authentication password (optional).
    """

    host: str = Field(default="localhost", description="Redis server host")
    port: int = Field(default=6379, description="Redis server port")
    db: int = Field(default=0, description="Redis database number", ge=0, le=15)
    password: SecretStr | None = Field(default=None, description="Redis password")

    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL."""
        password_part = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{password_part}{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class RabbitMQSettings(BaseSettings):
    """RabbitMQ configuration settings.

    Attributes:
        host: RabbitMQ server host.
        port: RabbitMQ server port.
        username: RabbitMQ username.
        password: RabbitMQ password.
        vhost: RabbitMQ virtual host.
    """

    host: str = Field(default="localhost", description="RabbitMQ server host")
    port: int = Field(default=5672, description="RabbitMQ server port", gt=0)
    username: str = Field(default="guest", description="RabbitMQ username")
    password: SecretStr = Field(default=SecretStr("guest"), description="RabbitMQ password")
    vhost: str = Field(default="/", description="RabbitMQ virtual host")

    @property
    def connection_url(self) -> str:
        """Generate RabbitMQ connection URL."""
        return (
            f"amqp://{self.username}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}{self.vhost}"
        )

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_RABBITMQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class DatabaseSettings(BaseSettings):
    """Database configuration settings.

    Attributes:
        host: Database server host.
        port: Database server port.
        name: Database name.
        user: Database user.
        password: Database password.
        pool_size: Connection pool size.
    """

    host: str = Field(default="localhost", description="Database server host")
    port: int = Field(default=5432, description="Database server port", gt=0)
    name: str = Field(default="agentarmy", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: SecretStr = Field(default=SecretStr("postgres"), description="Database password")
    pool_size: int = Field(default=20, description="Connection pool size", gt=0, le=100)

    @property
    def connection_url(self) -> str:
        """Generate database connection URL."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ClaudeSettings(BaseSettings):
    """Anthropic Claude LLM configuration settings.

    Attributes:
        api_key: Anthropic API key.
        default_model: Default Claude model to use.
        max_tokens: Maximum tokens for responses.
        timeout: Request timeout in seconds.
    """

    api_key: SecretStr = Field(default=SecretStr(""), description="Anthropic API key")
    default_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default Claude model",
    )
    max_tokens: int = Field(default=4096, description="Maximum tokens", gt=0, le=200000)
    timeout: float = Field(default=60.0, description="Request timeout in seconds", gt=0)

    @property
    def is_configured(self) -> bool:
        """Whether a real API key has been provided."""
        return bool(self.api_key.get_secret_value())

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_CLAUDE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class OpenAISettings(BaseSettings):
    """OpenAI LLM configuration settings.

    Attributes:
        api_key: OpenAI API key.
        default_model: Default OpenAI model to use.
    """

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model",
    )

    @property
    def is_configured(self) -> bool:
        """Whether a real API key has been provided."""
        return bool(self.api_key.get_secret_value())

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class GeminiSettings(BaseSettings):
    """Google Gemini LLM configuration settings.

    Attributes:
        api_key: Gemini API key.
        default_model: Default Gemini model to use.
    """

    api_key: SecretStr = Field(default=SecretStr(""), description="Gemini API key")
    default_model: str = Field(
        default="gemini-2.0-flash",
        description="Default Gemini model",
    )

    @property
    def is_configured(self) -> bool:
        """Whether a real API key has been provided."""
        return bool(self.api_key.get_secret_value())

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class KimiSettings(BaseSettings):
    """Kimi (Moonshot) LLM configuration settings.

    Attributes:
        api_key: Kimi/Moonshot API key.
        default_model: Default Kimi model to use.
    """

    api_key: SecretStr = Field(default=SecretStr(""), description="Kimi API key")
    default_model: str = Field(
        default="kimi-k2.5",
        description="Default Kimi model",
    )

    @property
    def is_configured(self) -> bool:
        """Whether a real API key has been provided."""
        return bool(self.api_key.get_secret_value())

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_KIMI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class Neo4jSettings(BaseSettings):
    """Neo4j knowledge graph configuration settings.

    Attributes:
        host: Neo4j server host.
        port: Neo4j Bolt protocol port.
        username: Neo4j authentication username.
        password: Neo4j authentication password.
        database: Target database name.
    """

    host: str = Field(default="localhost", description="Neo4j server host")
    port: int = Field(default=7687, description="Neo4j Bolt port", gt=0)
    username: str = Field(default="neo4j", description="Neo4j username")
    password: SecretStr = Field(default=SecretStr("neo4j"), description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")

    @property
    def bolt_url(self) -> str:
        """Generate Neo4j Bolt connection URL."""
        return f"bolt://{self.host}:{self.port}"

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_NEO4J_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class OllamaSettings(BaseSettings):
    """Ollama local LLM configuration settings.

    Attributes:
        base_url: Ollama server base URL.
        default_model: Default Ollama model to use.
    """

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )
    default_model: str = Field(default="llama2", description="Default Ollama model")

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class WhatsAppSettings(BaseSettings):
    """WhatsApp integration configuration settings.

    Attributes:
        api_token: WhatsApp API token.
        phone_number_id: WhatsApp phone number ID.
        verify_token: Webhook verification token.
        webhook_secret: Webhook secret for signing.
    """

    api_token: SecretStr = Field(default=SecretStr(""), description="WhatsApp API token")
    phone_number_id: str = Field(default="", description="WhatsApp phone number ID")
    verify_token: SecretStr = Field(default=SecretStr(""), description="WhatsApp webhook verify token")
    webhook_secret: SecretStr = Field(default=SecretStr(""), description="Webhook secret for signing")

    @property
    def is_configured(self) -> bool:
        """Whether WhatsApp credentials have been provided."""
        return bool(self.api_token.get_secret_value() and self.phone_number_id)

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_WHATSAPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class SecuritySettings(BaseSettings):
    """Security configuration settings.

    Attributes:
        jwt_secret: Secret key for JWT signing.
        jwt_algorithm: Algorithm for JWT encoding/decoding.
        token_expiry: Token expiration time in seconds.
        audit_hash_algorithm: Algorithm for audit log hashing.
    """

    jwt_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="JWT secret key (auto-generated if not set)",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm",
        pattern="^(HS256|HS512|RS256)$",
    )
    token_expiry: int = Field(
        default=3600, description="Token expiration in seconds", gt=0, le=86400
    )
    audit_hash_algorithm: str = Field(
        default="SHA256",
        description="Audit log hash algorithm",
        pattern="^(SHA256|SHA512)$",
    )

    @field_validator("jwt_algorithm", "audit_hash_algorithm", mode="before")
    @classmethod
    def uppercase_algorithms(cls, v: str) -> str:
        """Ensure algorithm names are uppercase."""
        return v.upper()

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ClusterSettings(BaseSettings):
    """Cluster configuration for multi-machine mode.

    Three modes:
        - standalone (default): Single local instance, no clustering.
        - center: Command Center on VPS/cloud, accepts worker connections via MCP.
        - worker: Local instance connecting to a remote Command Center.

    Attributes:
        mode: Cluster mode (standalone, center, worker).
        center_url: Command Center MCP server URL (worker mode only).
        center_token: Shared authentication token for worker ↔ center trust.
        worker_name: Human-readable name for this worker instance.
        worker_port: Port for the worker's local MCP server (receives dispatched tasks).
        heartbeat_interval: Seconds between worker heartbeat pings.
        stale_timeout: Seconds before center considers a worker offline.
    """

    mode: Literal["standalone", "center", "worker"] = Field(
        default="standalone", description="Cluster mode"
    )
    center_url: str = Field(
        default="", description="Command Center MCP URL (worker mode)"
    )
    center_token: SecretStr = Field(
        default=SecretStr(""), description="Shared cluster auth token"
    )
    worker_name: str = Field(
        default="", description="Human-readable worker name"
    )
    worker_port: int = Field(
        default=8002, description="Worker MCP server port", gt=0
    )
    heartbeat_interval: int = Field(
        default=30, description="Heartbeat interval in seconds", gt=5, le=300
    )
    stale_timeout: int = Field(
        default=90, description="Seconds before worker is considered offline", gt=10
    )

    @property
    def is_center(self) -> bool:
        return self.mode == "center"

    @property
    def is_worker(self) -> bool:
        return self.mode == "worker"

    @property
    def is_standalone(self) -> bool:
        return self.mode == "standalone"

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_CLUSTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class GoogleCalendarSettings(BaseSettings):
    """Google Calendar OAuth2 configuration.

    Required for OAuth2 flow — create a project in Google Cloud Console,
    enable the Google Calendar API, and create OAuth2 credentials.

    Attributes:
        oauth_client_id: OAuth2 client ID from Google Cloud Console.
        oauth_client_secret: OAuth2 client secret.
        oauth_redirect_uri: Callback URL (must be registered in GCP).
    """

    oauth_client_id: str = Field(
        default="", description="Google OAuth2 Client ID"
    )
    oauth_client_secret: SecretStr = Field(
        default=SecretStr(""), description="Google OAuth2 Client Secret"
    )
    oauth_redirect_uri: str = Field(
        default="http://localhost:8000/api/auth/google/callback",
        description="OAuth2 redirect/callback URI",
    )

    @property
    def is_configured(self) -> bool:
        """Check if OAuth2 credentials are set."""
        return bool(self.oauth_client_id and self.oauth_client_secret.get_secret_value())

    model_config = SettingsConfigDict(
        env_prefix="AGENTARMY_GOOGLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class MasterSettings(BaseSettings):
    """Master settings combining all configuration classes.

    Provides unified access to all subsystem configurations.
    """

    system: SystemSettings = Field(default_factory=SystemSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    kimi: KimiSettings = Field(default_factory=KimiSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    whatsapp: WhatsAppSettings = Field(default_factory=WhatsAppSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    google_calendar: GoogleCalendarSettings = Field(default_factory=GoogleCalendarSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_env(cls) -> "MasterSettings":
        """Load settings from environment variables and .env file.

        Returns:
            MasterSettings instance with all configuration loaded.
        """
        return cls(
            system=SystemSettings(),
            redis=RedisSettings(),
            rabbitmq=RabbitMQSettings(),
            database=DatabaseSettings(),
            claude=ClaudeSettings(),
            openai=OpenAISettings(),
            gemini=GeminiSettings(),
            kimi=KimiSettings(),
            neo4j=Neo4jSettings(),
            ollama=OllamaSettings(),
            whatsapp=WhatsAppSettings(),
            security=SecuritySettings(),
            cluster=ClusterSettings(),
            google_calendar=GoogleCalendarSettings(),
        )

    # ── Convenience properties used by main.py and agents ──────────

    @property
    def database_url(self) -> str:
        """PostgreSQL connection URL."""
        return self.database.connection_url

    @property
    def database_pool_size(self) -> int:
        """Database connection pool size."""
        return self.database.pool_size

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return self.redis.connection_url

    @property
    def redis_ttl(self) -> int:
        """Default Redis TTL in seconds."""
        return 3600

    @property
    def neo4j_bolt_url(self) -> str:
        """Neo4j Bolt connection URL."""
        return self.neo4j.bolt_url

    @property
    def neo4j_username(self) -> str:
        """Neo4j authentication username."""
        return self.neo4j.username

    @property
    def neo4j_password(self) -> str:
        """Neo4j authentication password (unwrapped)."""
        return self.neo4j.password.get_secret_value()

    @property
    def neo4j_database(self) -> str:
        """Neo4j database name."""
        return self.neo4j.database

    @property
    def whatsapp_enabled(self) -> bool:
        """Whether WhatsApp integration is enabled and configured."""
        import os
        env_flag = os.getenv("ENABLE_WHATSAPP", "true").lower() == "true"
        return env_flag and self.whatsapp.is_configured

    @property
    def whatsapp_api_token(self) -> str:
        """WhatsApp API token (unwrapped)."""
        return self.whatsapp.api_token.get_secret_value()

    @property
    def whatsapp_phone_number_id(self) -> str:
        """WhatsApp phone number ID."""
        return self.whatsapp.phone_number_id

    @property
    def whatsapp_verify_token(self) -> str:
        """WhatsApp webhook verification token (unwrapped)."""
        return self.whatsapp.verify_token.get_secret_value()

    @property
    def is_configured(self) -> bool:
        """Whether any LLM provider is configured (any API key or local Ollama)."""
        return (
            self.claude.is_configured
            or self.openai.is_configured
            or self.gemini.is_configured
            or self.kimi.is_configured
        )

    @property
    def active_llm_provider(self) -> str:
        """Return the name of the first configured LLM provider."""
        if self.claude.is_configured:
            return "claude"
        if self.openai.is_configured:
            return "openai"
        if self.gemini.is_configured:
            return "gemini"
        if self.kimi.is_configured:
            return "kimi"
        return "ollama"


# Alias so main.py can do: from src.core.config import Settings
Settings = MasterSettings
