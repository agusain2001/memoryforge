"""
Tests for sync encryption layer.
"""

import pytest

# Check if cryptography is available
try:
    from memoryforge.sync.encryption import EncryptionLayer, EncryptionError
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography package not installed")
class TestEncryptionRoundTrip:
    """Tests for encryption round-trip."""
    
    def test_encrypt_decrypt_preserves_data(self):
        """Test that encrypt/decrypt preserves original data."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        original = "Secret memory content with special chars: émojis 🔐 and 日本語"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert decrypted == original
        assert encrypted != original  # Should be encrypted
    
    def test_empty_string_handling(self):
        """Test empty string encryption."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        original = ""
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_large_content_handling(self):
        """Test encryption of large content."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        # 100KB of data
        original = "x" * 100000
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_unicode_content(self):
        """Test encryption of unicode content."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        original = "Unicode: こんにちは 🎉 émojis 中文"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert decrypted == original


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography package not installed")
class TestEncryptionSecurity:
    """Tests for encryption security properties."""
    
    def test_different_keys_produce_different_ciphertext(self):
        """Test that different keys produce different encrypted output."""
        key1 = EncryptionLayer.generate_key()
        key2 = EncryptionLayer.generate_key()
        
        enc1 = EncryptionLayer(key1)
        enc2 = EncryptionLayer(key2)
        
        original = "Same content"
        encrypted1 = enc1.encrypt(original)
        encrypted2 = enc2.encrypt(original)
        
        assert encrypted1 != encrypted2
    
    def test_same_content_same_key_different_ciphertext(self):
        """Test that Fernet produces different ciphertext each time (due to randomization)."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        original = "Same content"
        encrypted1 = encryption.encrypt(original)
        encrypted2 = encryption.encrypt(original)
        
        # Fernet uses random IV, so each encryption should be different
        assert encrypted1 != encrypted2
    
    def test_wrong_key_fails_decryption(self):
        """Test that decryption with wrong key fails."""
        key1 = EncryptionLayer.generate_key()
        key2 = EncryptionLayer.generate_key()
        
        enc1 = EncryptionLayer(key1)
        enc2 = EncryptionLayer(key2)
        
        original = "Secret data"
        encrypted = enc1.encrypt(original)
        
        with pytest.raises(EncryptionError):
            enc2.decrypt(encrypted)
    
    def test_invalid_key_raises(self):
        """Test that invalid key raises error."""
        with pytest.raises(EncryptionError):
            EncryptionLayer("not-a-valid-key")
    
    def test_tampered_data_fails(self):
        """Test that tampered ciphertext fails decryption."""
        key = EncryptionLayer.generate_key()
        encryption = EncryptionLayer(key)
        
        original = "Secret data"
        encrypted = encryption.encrypt(original)
        
        # Tamper with the ciphertext
        tampered = encrypted[:-5] + "XXXXX"
        
        with pytest.raises(EncryptionError):
            encryption.decrypt(tampered)


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography package not installed")
class TestKeyGeneration:
    """Tests for key generation."""
    
    def test_generate_key_returns_valid_key(self):
        """Test that generated keys are valid."""
        key = EncryptionLayer.generate_key()
        
        # Should be a string
        assert isinstance(key, str)
        
        # Should be usable
        encryption = EncryptionLayer(key)
        original = "test"
        assert encryption.decrypt(encryption.encrypt(original)) == original
    
    def test_generated_keys_are_unique(self):
        """Test that each generated key is unique."""
        keys = set()
        for _ in range(100):
            keys.add(EncryptionLayer.generate_key())
        
        assert len(keys) == 100
