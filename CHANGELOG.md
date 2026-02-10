# Changelog

All notable changes to MemoryForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-02-10

### Fixed
- **Critical:** `memoryforge sync push` crashed with `AttributeError: 'LocalFileAdapter' object has no attribute 'read'`
- **Critical:** `memoryforge sync pull` crashed with same adapter method name mismatch
- **Critical:** `memoryforge share memory` crashed with `AttributeError: 'LocalFileAdapter' object has no attribute 'write'`
- **Critical:** `memoryforge share list` crashed with `AttributeError: 'LocalFileAdapter' object has no attribute 'read'`
- **Critical:** `memoryforge share import` crashed with `AttributeError: 'LocalFileAdapter' object has no attribute 'exists'`
- `memoryforge sync push` and `sync pull` crashed when unpacking `SyncResult` as an integer instead of an object
- `memoryforge share memory` crashed with `NameError: name 'datetime' is not defined`

### Added
- Integration test suite with 8 tests covering full sync roundtrip scenarios
- Tests for encryption/decryption, export/import, conflict detection, force mode, project isolation, and idempotency
- `CHANGELOG.md` to track version history

### Changed
- README test badge updated from misleading "111 passed" to accurate "8 passed"
- Sync commands now display conflicts and errors when detected (instead of silent failure)

## [1.0.0] - 2026-02-09

### Added
- Initial stable release
- Multi-project support with project router
- Graph memory with relationship tracking
- Confidence scoring system
- Conflict resolution and logging
- Staleness tracking
- Git integration
- Team sync with encryption
- Memory consolidation with rollback
- MCP server for IDE integration
