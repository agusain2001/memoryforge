"""
Graph Builder for MemoryForge v3.

Builds read-only graph relationships between memories.
The graph is derived, not authoritative - SQLite remains the source of truth.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from memoryforge.models import Memory, MemoryRelation, RelationType
from memoryforge.storage.sqlite_db import SQLiteDatabase


class GraphBuilder:
    """Derives and manages memory-to-memory relationships."""
    
    def __init__(self, db: SQLiteDatabase):
        """Initialize the graph builder."""
        self.db = db
    
    def link_memories(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: RelationType,
        created_by: str = "human",
    ) -> MemoryRelation:
        """Create a link between two memories.
        
        Args:
            source_id: The source memory ID
            target_id: The target memory ID  
            relation_type: Type of relationship
            created_by: 'human' or 'git-derived'
        """
        # Validate both memories exist
        source = self.db.get_memory(source_id)
        target = self.db.get_memory(target_id)
        
        if not source:
            raise ValueError(f"Source memory {source_id} not found")
        if not target:
            raise ValueError(f"Target memory {target_id} not found")
        
        return self.db.create_memory_relation(
            source_memory_id=source_id,
            target_memory_id=target_id,
            relation_type=relation_type,
            created_by=created_by,
        )
    
    def get_graph_view(self, memory_id: UUID) -> dict:
        """Get a graph view centered on a memory.
        
        Returns a dict with:
        - memory: The central memory
        - incoming: Memories that link TO this memory
        - outgoing: Memories this memory links TO
        - causality_chain: Decision chain leading to this memory
        """
        memory = self.db.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        
        incoming_relations = self.db.get_memory_relations(memory_id, direction="incoming")
        outgoing_relations = self.db.get_memory_relations(memory_id, direction="outgoing")
        
        # Fetch the actual memories
        incoming_memories = []
        for rel in incoming_relations:
            mem = self.db.get_memory(rel.source_memory_id)
            if mem:
                incoming_memories.append({
                    "memory": mem,
                    "relation_type": rel.relation_type.value,
                })
        
        outgoing_memories = []
        for rel in outgoing_relations:
            mem = self.db.get_memory(rel.target_memory_id)
            if mem:
                outgoing_memories.append({
                    "memory": mem,
                    "relation_type": rel.relation_type.value,
                })
        
        causality_chain = self.db.get_causality_chain(memory_id)
        
        return {
            "memory": memory,
            "incoming": incoming_memories,
            "outgoing": outgoing_memories,
            "causality_chain": causality_chain,
        }
    
    def find_related_memories(
        self,
        memory_id: UUID,
        relation_types: Optional[list[RelationType]] = None,
        max_depth: int = 2,
    ) -> list[Memory]:
        """Find all memories related to a given memory up to a certain depth.
        
        Args:
            memory_id: Starting memory
            relation_types: Filter by these relation types (None = all)
            max_depth: Maximum relationship depth to traverse
        """
        visited: set[UUID] = set()
        result: list[Memory] = []
        current_level = {memory_id}
        
        for _ in range(max_depth):
            next_level: set[UUID] = set()
            
            for mid in current_level:
                if mid in visited:
                    continue
                visited.add(mid)
                
                relations = self.db.get_memory_relations(mid, direction="both")
                
                for rel in relations:
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    
                    # Get the other memory in the relation
                    other_id = (
                        rel.target_memory_id
                        if rel.source_memory_id == mid
                        else rel.source_memory_id
                    )
                    
                    if other_id not in visited:
                        next_level.add(other_id)
                        mem = self.db.get_memory(other_id)
                        if mem and mem not in result:
                            result.append(mem)
            
            current_level = next_level
        
        return result
    
    def unlink_memories(self, relation_id: UUID) -> bool:
        """Remove a link between memories."""
        return self.db.delete_memory_relation(relation_id)
    
    def get_decision_consequences(self, decision_id: UUID) -> list[Memory]:
        """Get all memories that were caused by a decision.
        
        Finds memories with 'caused_by' relation pointing to this decision.
        """
        relations = self.db.get_memory_relations(decision_id, direction="incoming")
        consequences = []
        
        for rel in relations:
            if rel.relation_type == RelationType.CAUSED_BY:
                mem = self.db.get_memory(rel.source_memory_id)
                if mem:
                    consequences.append(mem)
        
        return consequences
