import sys, os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # api目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录

# 创建FastAPI应用
app = FastAPI(
    title="DanceVibe API Test",
    version="3.0.0",
    description="舞蹈姿态检测和评分系统 - 测试版",
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

# 挂载静态文件
static_path = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    print(f"✅ 静态文件目录: {static_path}")
else:
    print(f"⚠️ 静态文件目录不存在: {static_path}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页路由 - 测试版"""
    print("🔍 根路由被调用了！")

    try:
        # 查找前端文件
        html_path = os.path.join(PROJECT_ROOT, "frontend", "dancevibe.html")
        print(f"🔍 查找前端文件: {html_path}")

        if os.path.exists(html_path):
            print("✅ 前端文件找到，返回HTML内容")
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            print("⚠️ 前端文件未找到，返回测试页面")
            return HTMLResponse(content=f'''
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>DanceVibe API 测试</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
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
                    .info {{
                        margin-top: 2rem;
                        font-size: 0.9rem;
                        opacity: 0.8;
                        text-align: left;
                        background: rgba(0, 0, 0, 0.2);
                        padding: 1rem;
                        border-radius: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎵 DanceVibe API 测试</h1>
                    <p class="status">✅ 服务器运行正常</p>
                    <p class="status">✅ 根路由正常工作</p>

                    <div class="links">
                        <a href="/docs" class="link">📖 API文档</a>
                        <a href="/redoc" class="link">📚 ReDoc文档</a>
                        <a href="/test" class="link">🧪 测试接口</a>
                    </div>

                    <div class="info">
                        <h3>📁 路径信息:</h3>
                        <p><strong>项目根目录:</strong> {PROJECT_ROOT}</p>
                        <p><strong>API目录:</strong> {BASE_DIR}</p>
                        <p><strong>前端文件路径:</strong> {html_path}</p>
                        <p><strong>前端文件存在:</strong> {os.path.exists(html_path)}</p>
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


@app.get("/test")
async def test_endpoint():
    """测试接口"""
    print("🧪 测试接口被调用")
    return {
        "message": "测试成功！",
        "status": "OK",
        "paths": {
            "project_root": PROJECT_ROOT,
            "api_dir": BASE_DIR,
            "frontend_exists": os.path.exists(os.path.join(PROJECT_ROOT, "frontend")),
            "html_exists": os.path.exists(os.path.join(PROJECT_ROOT, "frontend", "dancevibe.html"))
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "3.0.0-test",
        "paths": {
            "project_root": PROJECT_ROOT,
            "api_dir": BASE_DIR
        }
    }


if __name__ == '__main__':
    print("🚀 启动DanceVibe API测试服务器...")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"📁 API目录: {BASE_DIR}")
    print("📍 测试主页: http://localhost:5000")
    print("📍 API文档: http://localhost:5000/docs")
    print("📍 测试接口: http://localhost:5000/test")

    try:
        uvicorn.run(
            "test_app:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")