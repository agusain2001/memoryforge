"""
Tests for the validation layer.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from memoryforge.models import MemoryCreate, MemoryType, MemorySource
from memoryforge.core.validation import ValidationLayer, ValidationError


class TestValidationLayer:
    """Tests for input validation."""
    
    def test_valid_memory_create(self):
        """Test validation passes for valid input."""
        data = MemoryCreate(
            content="We use React with TypeScript",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
        )
        
        # Should not raise
        ValidationLayer.validate_memory_create(data)
    
    def test_empty_content_fails_at_pydantic(self):
        """Test validation fails for empty content at Pydantic level."""
        # Empty content violates min_length=1 in Pydantic model
        with pytest.raises(PydanticValidationError):
            MemoryCreate(
                content="",
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
            )
    
    def test_whitespace_only_content_fails(self):
        """Test validation fails for whitespace-only content."""
        # Whitespace content passes Pydantic min_length but fails our validation
        data = MemoryCreate(
            content="   \n\t   ",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            ValidationLayer.validate_memory_create(data)
        
        assert "empty" in exc_info.value.message.lower()
    
    def test_content_too_long_fails_at_pydantic(self):
        """Test validation fails for content exceeding max length at Pydantic level."""
        # Over 10KB limit violates max_length in Pydantic model
        with pytest.raises(PydanticValidationError):
            MemoryCreate(
                content="x" * 11000,
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
            )
    
    def test_sanitize_content(self):
        """Test content sanitization."""
        content = "  \r\n  Hello World  \r\n  "
        
        sanitized = ValidationLayer.sanitize_content(content)
        
        assert sanitized == "Hello World"
        assert "\r\n" not in sanitized
    
    def test_sanitize_removes_null_chars(self):
        """Test that null characters are removed."""
        content = "Hello\x00World"
        
        sanitized = ValidationLayer.sanitize_content(content)
        
        assert "\x00" not in sanitized
        assert sanitized == "HelloWorld"
    
    def test_validate_search_query_empty(self):
        """Test validation fails for empty search query."""
        with pytest.raises(ValidationError):
            ValidationLayer.validate_search_query("")
    
    def test_validate_search_query_valid(self):
        """Test validation passes for valid search query."""
        # Should not raise
        ValidationLayer.validate_search_query("What framework do we use?")


class TestMemoryTypes:
    """Tests for memory type validation."""
    
    def test_all_memory_types_valid(self):
        """Test all defined memory types are valid."""
        for mem_type in MemoryType:
            data = MemoryCreate(
                content="Test content",
                type=mem_type,
                source=MemorySource.MANUAL,
            )
            
            # Should not raise
            ValidationLayer.validate_memory_create(data)
    
    def test_all_memory_sources_valid(self):
        """Test all defined memory sources are valid."""
        for source in MemorySource:
            data = MemoryCreate(
                content="Test content",
                type=MemoryType.NOTE,
                source=source,
            )
            
            # Should not raise
            ValidationLayer.validate_memory_create(data)
