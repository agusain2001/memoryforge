"""
Encryption Layer for MemoryForge Sync.

Handles AES-256 encryption using cryptography.fernet.
Required for secure syncing of memories.
"""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Encryption/Decryption failure."""
    pass


class EncryptionLayer:
    """
    Handles encryption of memory content.
    
    Uses Fernet (symmetric encryption) which guarantees confidentiality 
    and integrity (HMAC).
    """
    
    def __init__(self, key: str):
        """
        Initialize encryption layer.
        
        Args:
            key: 32-byte URL-safe base64-encoded key
        """
        self._check_dependency()
        
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(key)
        except Exception as e:
            raise EncryptionError(f"Invalid encryption key: {e}")
    
    def _check_dependency(self) -> None:
        """Check if cryptography package is installed."""
        try:
            import cryptography
        except ImportError:
            raise ImportError(
                "cryptography package is required for sync encryption. "
                "Install it with: pip install memoryforge[sync]"
            )
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.
        
        Args:
            data: Plaintext data
            
        Returns:
            Encrypted string (base64)
        """
        if not data:
            return ""
            
        try:
            # Fernet encrypts bytes -> bytes
            encrypted_bytes = self._fernet.encrypt(data.encode("utf-8"))
            return encrypted_bytes.decode("utf-8")
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, token: str) -> str:
        """
        Decrypt string data.
        
        Args:
            token: Encrypted token (base64 string)
            
        Returns:
            Decrypted plaintext string
        """
        if not token:
            return ""
            
        try:
            # Fernet decrypts bytes -> bytes
            decrypted_bytes = self._fernet.decrypt(token.encode("utf-8"))
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new valid Fernet key."""
        try:
            from cryptography.fernet import Fernet
            return Fernet.generate_key().decode("utf-8")
        except ImportError:
            raise ImportError("cryptography package required to generate key")
