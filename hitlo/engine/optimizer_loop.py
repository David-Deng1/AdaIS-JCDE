# -*- coding: utf-8 -*-
"""后台优化线程"""

import time


def optimizer_loop(state):
    """后台优化线程主循环。

    在 state.running 为 True 时持续调用 state.step_opt()。
    """
    while getattr(state, "running", False):
        try:
            state.step_opt()
        except Exception as e:
            state.add_log(f"[ERR] step_opt: {e}")
        time.sleep(0.15)
