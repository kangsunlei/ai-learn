# Qwen function call Demo

## 项目结构

```
qwen_fc_demo/
├── .env               # 环境变量（包含API密钥）
├── pyproject.toml     # 项目配置和依赖
├── src/               # 源代码目录
│   └── paper.py        # arXiv论文检索与分析工具
```

## 安装步骤

### 1. 安装 uv

如果您尚未安装 uv，请按照以下步骤进行安装：

```bash
# 在 macOS 上使用 Homebrew 安装
brew install uv

# 或者使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建虚拟环境

使用 uv 创建虚拟环境：

```bash
# 创建虚拟环境
uv venv .venv
```

### 3. 激活虚拟环境

```bash
# 在 macOS/Linux 上:
source .venv/bin/activate

# 在 Windows 上:
.venv\Scripts\activate
```

### 4. 安装依赖

激活虚拟环境后，使用 uv 安装项目依赖：

```bash
# 安装依赖
uv install -e .
```

### 5. 配置 API 密钥

编辑`.env`文件，添加您的 DashScope API 密钥:

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

> 注意：您需要从阿里云 DashScope 获取 API 密钥。

## 功能模块

### 论文检索与分析

使用 arxiv 库检索最新的学术论文，并利用千问大模型进行解析和分析。

```bash
cd src
python paper.py
```

这个模块提供了以下功能：

- 按主题搜索 arXiv 论文
- 保存论文元信息到本地 JSON 文件
- 使用千问大模型分析论文内容

### arXiv 论文检索工具

paper.py 实现了以下功能:

- **search_papers(topic, max_results)**: 根据主题在 arXiv 上搜索论文，返回论文 ID 列表
- **extract_info(paper_id)**: 获取指定论文 ID 的详细信息
- **chat_loop()**: 交互式查询论文信息的命令行界面

#### 论文存储结构

论文信息以 JSON 格式存储在`papers/`目录中，按主题分类：

```

papers/
└── {topic}/
└── papers_info.json

```

每个论文条目包含标题、作者、摘要、PDF 链接和发布日期等信息。

## 依赖项

项目使用了以下主要依赖：

- **python-dotenv**: 用于加载环境变量
- **openai**: 用于与符合 OpenAI 接口的 API 进行通信
- **arxiv**: 用于检索 arXiv 论文信息
- **dashscope**: 阿里云千问 API 客户端
- **feedparser**: arxiv 库的依赖，用于解析 RSS feed
- **requests**: HTTP 请求库

完整依赖列表请参见`pyproject.toml`文件。

## 开发说明

关于最新的开发进度和计划，请参阅`CHANGELOG.md`文件。

```

```
