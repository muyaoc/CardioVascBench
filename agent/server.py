import uvicorn

# 应用工厂和日志配置
from medical_graph_api.app_factory import create_app
from medical_graph_api.logging_setup import setup_logging

# 初始化日志和应用
setup_logging()
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有网络接口
        port=8801,       # 服务端口
        limit_concurrency=20,     # 最大并发连接数
        timeout_keep_alive=60,    # 连接保持超时(秒)
    )