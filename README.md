# 微博舆情管理系统

> 内网单机部署的微博舆情采集、趋势、人工标注与情感引擎评估系统。  
> 产品设计：[`docs/10_产品设计文档.md`](docs/10_产品设计文档.md)  
> 系统架构：[`docs/11_系统架构设计.md`](docs/11_系统架构设计.md)

## 这个系统解决什么

小团队需要持续监控微博上的声量和情感走势，及时发现负面事件，并人工标注一批样本校准自动情感引擎。本系统提供：

- **采集**：关键词搜索、命中帖评论/转发、指定账号时间线、热搜榜；账号池、2.5 秒限速、418/403 冷却、Cookie 加密。
- **存储**：PostgreSQL 16；任务队列与情感队列都用 PG 表 + `FOR UPDATE SKIP LOCKED`，不引入 Redis/MQ。
- **分析**：声量、情感占比、负面突增、高互动负面帖、热搜命中。
- **标注**：多人协作、一人一条可修改、多数票 + 最新票打平的最终情感规则。
- **评估**：本地 / LLM 引擎可切换，按人工标注计算准确率、各类 F1、混淆矩阵。
- **前端**：Vue3 + Element Plus + ECharts，四个页面：舆情列表、趋势看板、标注工作台、设置。

## 架构一览

```text
web (Vue3 + nginx) ──/api──> api (FastAPI + JWT)
                                  │
                                  ├── PostgreSQL 16
                                  │     ├ 业务表
                                  │     ├ sentiment_queue
                                  │     └ v_sentiment_final
                                  │
crawler (APScheduler) ────────────┘
  ├ 关键词/账号/热搜/评论/转发
  ├ AccountPool + 限速 + 冷却
  └ sentiment worker
```

详细模块、数据流、ER 图、API 表、部署与差距见 [`docs/11_系统架构设计.md`](docs/11_系统架构设计.md)。

## 快速开始

### 方式一：Docker Compose（推荐，需 Docker）

```bash
cd deploy
cp .env.example .env
# 修改 .env 里的数据库密码、JWT_SECRET、ADMIN_PASSWORD
docker compose up -d --build
```

- 前端：<http://localhost:8080>
- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>

默认 `CRAWLER_MODE=mock`，只读录制样本，不访问微博。

### 方式二：本地 SQLite（无需 Docker）

```bash
cd backend
python -m pip install -r requirements-dev.txt

# 设置演示管理员密码；Windows PowerShell 用：$env:ADMIN_PASSWORD="admin123"
export ADMIN_PASSWORD=admin123

# 导入 1,000 条公开数据样本并生成自动情感结果
python -m app.seed --csv ../data/crawl/weibo_1000.csv

uvicorn app.main:app --reload --port 8000
```

另开终端：

```bash
cd frontend
npm install
npm run dev
```

Vite 会把 `/api` 代理到 `127.0.0.1:8000`。

### 演示账号

| 角色 | 用户名 | 密码 |
|---|---|---|
| 管理员 | `admin` | `ADMIN_PASSWORD`；未设置时首次启动随机生成并打印在日志 |
| 标注员 | `annotator` | `annotator123`（仅演示数据导入时创建） |

生产环境请修改密码或删除演示账号。若未设置 `ADMIN_PASSWORD`，请在后端日志中查找随机密码。

## 主要命令

```bash
# 后端测试
cd backend && python -m pytest tests -q

# 采集一轮（mock 默认；http 需先在设置页导入 Cookie）
python -m crawler.scheduler --once

# 常驻调度
python -m crawler.scheduler

# 消费情感队列
python -m crawler.worker --once

# 前端构建
cd frontend && npm run build
```

## 目录结构

```text
backend/
  app/               FastAPI、SQLAlchemy、鉴权、业务服务、种子数据
  crawler/           微博客户端、解析器、账号池、任务、调度器、worker、fixtures
  tests/             API/解析/账号池/队列测试
frontend/
  src/views/         舆情列表、趋势看板、标注工作台、设置
  src/layout/        侧边栏布局
  src/api.js         Axios + JWT
  src/router.js      Hash 路由 + 登录/管理员守卫
deploy/
  docker-compose.yml db/api/crawler/web
  nginx.conf         静态托管 + /api 反代
  .env.example       密钥与模式模板
docs/
  10_产品设计文档.md
  11_系统架构设计.md
src/                 历史离线分析流水线（保留）
dashboard/           历史离线大屏（保留）
data/crawl/          1,000 条匿名公开数据样本（演示）
```

## 设计边界与当前状态

已完成并在本机验证：

- FastAPI + JWT + RBAC、帖子/趋势/热搜/标注/评估/配置 API；
- SQLite 建表与 `v_sentiment_final` 视图，11 个 pytest 通过；
- Vue3 四页面，`npm run build` 通过；
- Mock 采集、解析器、账号池冷却、情感队列与 worker；
- 1,000 条公开数据演示导入 + 本地 TF-IDF+LR 自动打分。

需要环境支持、未在本机实际运行：

- Docker Compose 与 PostgreSQL 16（本机未安装 Docker）；
- 真实微博 HTTP 采集（需要管理员提供 Cookie）；
- 扫码登录（返回 501，当前用手动 Cookie 导入）；
- LLM 引擎（需要配置 OpenAI 兼容 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`）。

诚实说明：

- `local` 引擎当前是文本 TF-IDF+LR，文件缺失时退化为词典；**不是中文 RoBERTa**。
- 演示数据来自公开数据集，不是实时爬取；不伪造人工标注，评估页在未标注时显示 0 条。
- 告警推送、分布式爬虫、代理池、多平台不在本期范围。

## 文档

| 文档 | 内容 |
|---|---|
| [产品设计文档](docs/10_产品设计文档.md) | 目标、非目标、数据模型、API、页面、部署 |
| [系统架构设计](docs/11_系统架构设计.md) | 拓扑、分层、数据流、ER 图、最终情感、部署、测试、差距 |
| [历史研究报告](docs/01_研究报告.md) | 原离线多模态分析作品（保留） |
| [历史开发文档](docs/02_开发文档.md) | 原离线流水线与实验（保留） |

## 合规红线

- 只采集公开内容，不采集私信/非公开主页；
- 不实现登录绕过、验证码破解、代理池；
- Cookie 加密存储，仅内网登录用户可用；
- 单账号请求间隔 ≥ 2.5 秒，418/403 冷却；
- 原始 JSON 默认保留 90 天，定时清理；
- 不把公开数据演示样本冒充实时微博。
