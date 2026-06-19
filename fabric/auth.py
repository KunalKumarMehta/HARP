"""
HARP — Fabric authentication module  ·  MIT

WSS + bearer token authentication for the WebSocket fabric.
This is the "production one-liner" from ADR-010: wrap the raw WS in WSS + token.

Design:
- Token is a pre-shared secret (configured via env on both sides)
- Client sends Authorization: Bearer <token> on connect
- Server validates token before accepting the connection
- TLS context is configurable; self-signed certs work for dev, proper certs for prod
"""

from __future__ import annotations

import hmac
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AuthConfig:
    """Authentication + TLS configuration for a fabric endpoint."""
    token: Optional[str] = None                    # Bearer token (or from env)
    tls_enabled: bool = False                       # Use WSS instead of WS
    cert_file: Optional[str] = None                 # Path to certificate file (PEM)
    key_file: Optional[str] = None                  # Path to private key file (PEM)
    ca_file: Optional[str] = None                   # Path to CA bundle for client cert verification
    verify_hostname: bool = True                    # Verify server hostname (client side)

    @classmethod
    def from_env(cls, prefix: str = "HARP_FABRIC") -> "AuthConfig":
        """Load config from environment variables.
        
        Env vars:
            {prefix}_TOKEN        - Bearer token for auth
            {prefix}_TLS          - "1"/"true" to enable WSS
            {prefix}_CERT         - Path to cert file
            {prefix}_KEY          - Path to key file
            {prefix}_CA           - Path to CA bundle
            {prefix}_VERIFY_HOST  - "0"/"false" to disable hostname verification
        """
        token = os.getenv(f"{prefix}_TOKEN")
        tls_enabled = os.getenv(f"{prefix}_TLS", "0").lower() in ("1", "true", "yes")
        cert_file = os.getenv(f"{prefix}_CERT")
        key_file = os.getenv(f"{prefix}_KEY")
        ca_file = os.getenv(f"{prefix}_CA")
        verify_hostname = os.getenv(f"{prefix}_VERIFY_HOST", "1").lower() not in ("0", "false", "no")
        return cls(
            token=token,
            tls_enabled=tls_enabled,
            cert_file=cert_file,
            key_file=key_file,
            ca_file=ca_file,
            verify_hostname=verify_hostname,
        )

    def get_server_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for a WSS server (the side that serves)."""
        if not self.tls_enabled or not self.cert_file or not self.key_file:
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert_file, self.key_file)
        # Optional: require client certs
        if self.ca_file:
            ctx.load_verify_locations(self.ca_file)
            ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx

    def get_client_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for a WSS client (the side that connects).

        - ca_file given        -> verify the server cert against that CA (secure;
                                  use this to trust a self-signed/dev cert by
                                  pointing it at the cert itself).
        - HARP_FABRIC_VERIFY_HOST=0 (verify_hostname=False) -> INSECURE dev opt-out:
                                  accept ANY cert without verification. Never in prod.
        - default              -> secure context verifying against system CAs. A
                                  self-signed server cert WILL be rejected here; set
                                  a CA or verify_host=0 for dev self-signed.
        """
        if not self.tls_enabled:
            return None
        ctx = ssl.create_default_context()
        if self.ca_file:
            ctx.load_verify_locations(self.ca_file)
            return ctx
        if not self.verify_hostname:
            ctx.check_hostname = False          # must clear before CERT_NONE
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def validate_token(self, provided: Optional[str]) -> bool:
        """Validate a provided bearer token. Constant-time to avoid leaking the
        token via comparison timing."""
        if self.token is None:
            return True                          # No token configured = auth disabled
        if provided is None:
            return False
        return hmac.compare_digest(provided, self.token)

    def extract_token_from_headers(self, headers) -> Optional[str]:
        """Extract Bearer token from WebSocket handshake headers.
        
        Handles both websockets 14.x (list of tuples) and 15.x (Headers object) formats.
        """
        # Try Headers.items() / Headers.raw_items() (websockets 15.x)
        if hasattr(headers, 'items'):
            try:
                auth_header = headers.get('Authorization') or headers.get('authorization')
                if auth_header and auth_header.startswith("Bearer "):
                    return auth_header[7:]
            except Exception:
                pass
            # Fallback: iterate raw_items
            try:
                for name, value in headers.raw_items():
                    if name.lower() == "authorization" and value.startswith("Bearer "):
                        return value[7:]
            except Exception:
                pass
            # Old-style iter() over values
            try:
                for name, value in headers.items():
                    if name.lower() == "authorization" and value.startswith("Bearer "):
                        return value[7:]
            except Exception:
                pass
        
        # Legacy: list of (name, value) tuples (websockets 14.x or custom)
        try:
            for name, value in headers:
                if name.lower() == "authorization" and value.startswith("Bearer "):
                    return value[7:]
        except Exception:
            pass
        return None


# Convenience: global default config loaded from env at import time
DEFAULT_AUTH = AuthConfig.from_env()


def create_self_signed_cert(cert_path: str, key_path: str, *, bits: int = 2048, days: int = 365) -> None:
    """Generate a self-signed certificate for development/testing.
    
    Requires `cryptography` package (optional dependency).
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
    except ImportError:
        raise RuntimeError("cryptography package required: pip install cryptography")

    key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "HARP Fabric Dev"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    Path(cert_path).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    Path(key_path).write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )