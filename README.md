
# Text Moderation Server

## Installation

```bash
# 创建虚拟环境（推荐）
python -m venv venv
# Windows激活虚拟环境
venv\Scripts\activate
# Linux/Mac激活虚拟环境
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 运行服务器

```bash
# 方法1：使用启动脚本
python start.py

# 方法2：直接使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 使用CUDA（如果可用）
设置环境变量 `USE_CUDA=true` 启用GPU加速：

```bash
# Windows
set USE_CUDA=true
python start.py

# Linux/Mac
# export USE_CUDA=true
# python start.py
```

## Endpoint

POST `/text-moderate`

### Payload
```json
{
  "input": "This is a test text"
}
```

### Response
```json
{
  "result": [
    {
      "label": "not_offensive",
      "score": 0.95
    }
  ]
}

