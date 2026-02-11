"""日志初始化。

为了让各模块能够统一输出结构化日志，本项目在启动时会调用 `setup_logging()`：
- 设置 logging 的全局 level 与输出格式
- 返回一个约定名称的 logger（"MedicalGraphAPI"）

注意：这里使用的是 `logging.basicConfig`，多次调用在大多数情况下只会生效一次。
"""

import logging


def setup_logging() -> logging.Logger:
    """初始化全局日志配置并返回项目 logger。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("MedicalGraphAPI")
