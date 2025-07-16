import sys, os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 导入所有路由和WebSocket处理器
from game_router import router as game_router
from video_router import router as video_router
from pose_router import router as pose_router
from audio_router import router as audio_router
from scoring_router import router as scoring_router
from status_router import router as status_router
from ws_handler import websocket_router

# 设置路径 - 修正路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # api目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录

# 创建FastAPI应用
app = FastAPI(
    title="DanceVibe API",
    version="3.0.0",
    description="舞蹈姿态检测和评分系统",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件 - 修正静态文件路径
static_path = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    print(f"✅ 静态文件目录: {static_path}")
else:
    print(f"⚠️ 静态文件目录不存在: {static_path}")

# 注册所有路由
app.include_router(game_router, prefix="/api/game", tags=["游戏管理"])
app.include_router(video_router, prefix="/api/video", tags=["视频处理"])
app.include_router(pose_router, prefix="/api/pose", tags=["姿态检测"])
app.include_router(audio_router, prefix="/api/audio", tags=["音频分析"])
app.include_router(scoring_router, prefix="/api/scoring", tags=["评分系统"])
app.include_router(status_router, prefix="/api/status", tags=["状态查询"])
app.include_router(websocket_router, prefix="", tags=["WebSocket"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页路由"""
    try:
        # 修正前端文件路径
        html_path = os.path.join(PROJECT_ROOT, "frontend", "dancevibe.html")
        print(f"🔍 查找前端文件: {html_path}")

        if os.path.exists(html_path):
            print("✅ 前端文件找到，返回HTML内容")
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            print("⚠️ 前端文件未找到，返回临时页面")
            # 返回临时的简单HTML页面
            return HTMLResponse(content=f'''
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>DanceVibe API v3.0</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        margin: 0;
                        padding: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        color: white;
                    }}
                    .container {{
                        text-align: center;
                        background: rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        padding: 3rem;
                        border-radius: 20px;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    }}
                    h1 {{
                        font-size: 3rem;
                        margin-bottom: 1rem;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    }}
                    .links {{
                        margin-top: 2rem;
                    }}
                    .link {{
                        display: inline-block;
                        margin: 0.5rem 1rem;
                        padding: 1rem 2rem;
                        background: rgba(255, 255, 255, 0.2);
                        color: white;
                        text-decoration: none;
                        border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        transition: all 0.3s ease;
                    }}
                    .link:hover {{
                        background: rgba(255, 255, 255, 0.3);
                        transform: translateY(-2px);
                    }}
                    .status {{
                        margin-top: 2rem;
                        font-size: 1.1rem;
                        opacity: 0.9;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎵 DanceVibe API v3.0</h1>
                    <p class="status">✅ 服务器运行正常</p>
                    <p>前端文件路径: <code>{html_path}</code></p>
                    <p>项目根目录: <code>{PROJECT_ROOT}</code></p>

                    <div class="links">
                        <a href="/docs" class="link">📖 API文档</a>
                        <a href="/redoc" class="link">📚 ReDoc文档</a>
                        <a href="/api/status/health" class="link">🔍 健康检查</a>
                    </div>

                    <div style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.8;">
                        <p>💡 要使用完整前端界面，请将 <strong>dancevibe.html</strong> 文件放在:</p>
                        <p><code>{os.path.join(PROJECT_ROOT, "frontend")}</code> 目录下</p>
                    </div>
                </div>
            </body>
            </html>
            ''', status_code=200)

    except Exception as e:
        print(f"❌ 主页路由错误: {e}")
        return HTMLResponse(content=f'''
        <h1>🚨 错误</h1>
        <p>加载页面失败: {e}</p>
        <p><a href="/docs">查看API文档</a></p>
        ''', status_code=500)


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("🚀 DanceVibe API v3.0 启动完成")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"📁 API目录: {BASE_DIR}")
    print("📍 API文档: http://localhost:5000/docs")
    print("🔗 WebSocket: ws://localhost:5000/ws")
    print("🏠 主页: http://localhost:5000")


if __name__ == '__main__':
    print("🚀 启动DanceVibe API服务器...")
    try:
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=5000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")