# -*- coding: utf-8 -*-
"""HITLO 优化引擎模块"""

from .metrics import is_nondominated, crowding_distance, select_representative_indices
from .optimizer_loop import optimizer_loop

# OptState 延迟导入以避免循环依赖
def get_opt_state_class():
    from .opt_state import OptState
    return OptState
