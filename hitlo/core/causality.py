# -*- coding: utf-8 -*-
"""
hitlo.core.causality
======================
Decision causal graph for Ch6 §6.2 replay / traceability.

Stores DecisionNodes keyed by tick, enabling:
- Forward traversal (what happened after event X)
- Backward traceability (why was this action taken)
- Snapshot-at-tick (reconstruct UPV/OSV/metrics state at any tick)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionNode:
    """A single node in the decision causal graph."""
    node_id: int
    tick: int
    event_type: str  # preference_input | upv_update | osv_action | algo_dispatch | metric_change
    source_ids: List[int] = field(default_factory=list)  # parent node IDs
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CausalGraph:
    """Thread-safe causal graph for optimization decision tracking.

    Usage::

        graph = CausalGraph()
        n1 = graph.add_node(tick=1, event_type="preference_input",
                            payload={"channel": "slider", "w": [0.6, 0.4]})
        n2 = graph.add_node(tick=1, event_type="upv_update",
                            source_ids=[n1],
                            payload={"w": [0.6, 0.4], "tau": 0.2})
    """

    def __init__(self):
        self._nodes: List[DecisionNode] = []
        self._lock = threading.Lock()
        self._counter = 0
        self._snapshots: Dict[int, Dict[str, Any]] = {}  # tick -> state snapshot

    def add_node(
        self,
        tick: int,
        event_type: str,
        source_ids: Optional[List[int]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a decision node. Returns the node_id."""
        with self._lock:
            nid = self._counter
            self._counter += 1
            node = DecisionNode(
                node_id=nid,
                tick=tick,
                event_type=event_type,
                source_ids=source_ids or [],
                payload=payload or {},
            )
            self._nodes.append(node)
        return nid

    def save_snapshot(self, tick: int, snapshot: Dict[str, Any]):
        """Save a state snapshot at a given tick for later replay."""
        with self._lock:
            self._snapshots[tick] = snapshot

    def snapshot_at_tick(self, tick: int) -> Optional[Dict[str, Any]]:
        """Return the saved state snapshot at a given tick, or None."""
        with self._lock:
            return self._snapshots.get(tick)

    def trace_back(self, node_id: int) -> List[DecisionNode]:
        """Trace backward from a node to all its ancestors (BFS)."""
        with self._lock:
            id_map = {n.node_id: n for n in self._nodes}
            if node_id not in id_map:
                return []
            visited = set()
            queue = [node_id]
            result = []
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                node = id_map.get(nid)
                if node is None:
                    continue
                result.append(node)
                queue.extend(node.source_ids)
            return result

    def nodes_at_tick(self, tick: int) -> List[DecisionNode]:
        """Return all nodes at a specific tick."""
        with self._lock:
            return [n for n in self._nodes if n.tick == tick]

    def nodes_by_type(self, event_type: str) -> List[DecisionNode]:
        """Return all nodes of a specific event type."""
        with self._lock:
            return [n for n in self._nodes if n.event_type == event_type]

    def all_nodes(self) -> List[DecisionNode]:
        """Return a copy of all nodes."""
        with self._lock:
            return list(self._nodes)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def tick_range(self) -> tuple:
        """Return (min_tick, max_tick) or (0, 0) if empty."""
        with self._lock:
            if not self._nodes:
                return (0, 0)
            ticks = [n.tick for n in self._nodes]
            return (min(ticks), max(ticks))

    @property
    def available_snapshots(self) -> List[int]:
        """Return sorted list of ticks that have saved snapshots."""
        with self._lock:
            return sorted(self._snapshots.keys())

    def clear(self):
        with self._lock:
            self._nodes.clear()
            self._snapshots.clear()
            self._counter = 0
