# Tushare Token 管理器

一个用于管理 tushare API token 的 Python 工具，支持 token 缓存、验证和自动获取功能。

## 功能特性

- 🔄 **自动 Token 管理**: 支持从 API 自动获取最新 token
- 💾 **本地缓存**: 将 token 缓存到本地，避免重复请求
- ✅ **Token 验证**: 自动验证 token 有效性
- 📊 **使用统计**: 显示 token 调用次数和限制信息

## 快速开始

### 1. 配置 API 提取码

编辑 `token_manager.py` 文件，设置您的 API 提取码：

```python
self.api_code = "你的提取码"  # 替换为您的实际提取码
```

### 2. 基本使用

```python
from token_manager import get_valid_token
import tushare as ts

# 获取有效的 token
token = get_valid_token()
if token:
    # 初始化 tushare
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 使用 API
    df = pro.index_daily(ts_code="000001.SH", start_date="20250101", end_date="20250102")
    print(df.head())
```

### 3. 输出示例：
```
=== tushare 基本使用示例 ===

1. 获取有效的 API Token...
✓ 从缓存读取到Token，正在验证...
✓ Token验证成功

1. 初始化 Pro API...
✓ Pro API 初始化成功

1. 获取上证指数日线数据...
✓ 获取到 5 条指数数据
```

