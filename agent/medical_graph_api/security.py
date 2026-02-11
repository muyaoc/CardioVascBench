"""鉴权相关逻辑。

当前服务使用简单的 Bearer Token 鉴权：
- 客户端需在请求头携带：`Authorization: Bearer <token>`
- token 与 `config.SERVER_ACCESS_TOKEN` 不一致则返回 401

说明：这是最简化的鉴权方案，适用于内网/测试环境。
生产环境建议接入更完整的认证与权限体系。
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import SERVER_ACCESS_TOKEN

# FastAPI 内置的 Bearer Token 解析器
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """校验请求中的 Bearer Token。

    Args:
        credentials: FastAPI 从请求头解析得到的认证信息。

    Returns:
        str: token 字符串。

    Raises:
        HTTPException: token 不正确时抛出 401。
    """
    if credentials.credentials != SERVER_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials
