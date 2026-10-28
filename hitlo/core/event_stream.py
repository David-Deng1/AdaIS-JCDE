# -*- coding: utf-8 -*-
"""
hitlo.core.event_stream
========================
统一交互事件流，对应Ch3 §3.2(5)定义的事件模型。

每个交互事件 e_k = (τ_k, o_k, r_k, c_k, μ_k) 记录：
- channel (τ): 交互通道类型
- target (o): 交互对象标识
- content (r): 反馈内容
- context (c): 当前优化过程上下文
- meta (μ): 元数据（时长、修订次数、置信度等）
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InteractionEvent:
    """A single interaction event in the unified event stream."""

    timestamp: float                              # 事件时间戳
    channel: str                                  # τ: "brush_roi" | "slider" | "ranking" | "pairwise" | "nl_dialogue"
    target: str                                   # o: 交互对象标识（如 "w1", "scatter", "pcp"）
    content: Dict[str, Any] = field(default_factory=dict)  # r: 反馈内容
    context: Dict[str, Any] = field(default_factory=dict)  # c: {n_eval, hv, cv_avg, div, n_nd}
    meta: Dict[str, Any] = field(default_factory=dict)     # μ: {duration, revisions, confidence}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "target": self.target,
            "content": self.content,
            "context": self.context,
            "meta": self.meta,
        }


class EventStream:
    """Thread-safe event stream collector.

    Usage::

        stream = EventStream()
        stream.push("slider", "w1", {"value": 0.7})
        stream.push("nl_dialogue", "chat", {"text": "提高f2权重"})
        recent = stream.recent(10)
    """

    def __init__(self, max_events: int = 10000):
        self._events: List[InteractionEvent] = []
        self._lock = threading.Lock()
        self._max = max_events

    def push(
        self,
        channel: str,
        target: str,
        content: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> InteractionEvent:
        """Record a new interaction event and return it."""
        evt = InteractionEvent(
            timestamp=time.time(),
            channel=channel,
            target=target,
            content=content or {},
            context=context or {},
            meta=meta or {},
        )
        with self._lock:
            self._events.append(evt)
            if len(self._events) > self._max:
                self._events = self._events[-self._max:]
        return evt

    def recent(self, n: int = 20) -> List[InteractionEvent]:
        """Return the most recent *n* events."""
        with self._lock:
            return list(self._events[-n:])

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def all_events(self) -> List[InteractionEvent]:
        with self._lock:
            return list(self._events)

    def by_channel(self, channel: str) -> List[InteractionEvent]:
        """Return all events for a specific channel."""
        with self._lock:
            return [e for e in self._events if e.channel == channel]

    def clear(self):
        with self._lock:
            self._events.clear()
