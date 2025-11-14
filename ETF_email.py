import pandas as pd
import tushare as ts
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
from dotenv import load_dotenv
# from tushare_token_manager.token_manager import get_valid_token

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== ETF映射字典 =====
etf_dict = {
    "国内指数": {
        "上证50": "510050.SH",
        "沪深300": "510300.SH",
        "中证1000": "512100.SH",
        "创业板": "159915.SZ",
        "科创50": "588080.SH",
    },
    "国际指数": {
        "纳指": "513100.SH",
        "德国30": "513030.SH",
        "法国40": "513080.SH",
        "日经": "513520.SH",
        "沙特": "520830.SH",
        "港股通互联网": "159792.SZ",
        "港股创新药": "159567.SZ",
        "东南亚科技": "513730.SH",
        "中概互联": "513220.SH",
    },
    "行业题材": {
        "芯片": "588200.SH",
        "人工智能F": "515070.SH",
        "机器人": "562500.SH",
        "软件服务": "159852.SZ",
        "动漫游戏": "159869.SZ",
        "车电池": "159755.SZ",
        "证券公司": "512000.SH",
        "银行": "512800.SH",
        "房地产": "512200.SH",
        "有色": "512400.SH",
        "化工": "159870.SZ",
        "煤炭": "515220.SH",
        "稀土": "516150.SH",
        "消费": "159928.SZ",
        "白酒": "161725.SZ",
        "医疗": "512170.SH",
        "畜牧业": "159865.SZ",
        "军工": "512660.SH",
        "中药": "560080.SH",
        "旅游": "159766.SZ",
        "食品饮料": "159736.SZ",
        # 新添加
        "煤炭": "515200.SH",
        "新能源": "516850.SH",
        "化工": "159870.SZ",
        "机械": "159886.SZ",
        "新能源": "516160.SH",
        "家电": "159996.SZ",
        "房地产": "512200.SH",
        "传媒": "512980.SH",
        "5G通信": "515050.SH",
        "医药": "512010.SH",
        "医疗器械": "562600.SH",  
        "人工智能": "515980.SH",
        "影视": "159855.SZ",
        "电力": "159611.SZ"
    },
    "大宗商品": {
        "黄金": "518880.SH",
        "原油": "513690.SH"
    }
}


# ===== 添加扁平化工具函数 =====
def flatten_etf_dict(d):
    """
    将嵌套的 etf_dict（group -> {name: code}）扁平化为 {name: code}。
    同时返回 code -> group 的映射，用于分类显示。
    """
    flat = {}
    code_to_group = {}
    for group_name, group_dict in d.items():
        if isinstance(group_dict, dict):
            for name, code in group_dict.items():
                flat[name] = code
                code_to_group[code] = group_name
        else:
            # 扁平情况下直接处理
            flat[group_name] = group_dict
            code_to_group[group_dict] = ""
    return flat, code_to_group


# 全局生成扁平映射和分组映射
flat_etf, code_to_group = flatten_etf_dict(etf_dict)
flat_codes = list(flat_etf.values())


def update_etf_data(csv_path="./data/etf.csv", token=None, timeout=300, per_code_retries=3, retry_delay=5):
    """
    使用 tushare 接口更新ETF日行情数据，并保存到本地

    新增参数（超时/重试）：
        timeout: 整个更新过程的最长允许秒数（默认300秒）
        per_code_retries: 每个 ts_code 的最大重试次数（默认3）
        retry_delay: 每次重试之间的等待秒数（默认5秒）
    """
    if token is None:
        raise ValueError("请传入 tushare token")

    pro = ts.pro_api(token)

    if os.path.exists(csv_path):
        df_local = pd.read_csv(csv_path, dtype={"ts_code": str, "trade_date": str})
    else:
        df_local = pd.DataFrame()

    all_data = []
    start_time = time.time()

    # 使用扁平化后的代码列表
    for ts_code in flat_codes:
        # 检查总超时
        elapsed_total = time.time() - start_time
        if elapsed_total > timeout:
            raise TimeoutError(f"更新超时（超过 {timeout} 秒），在处理 {ts_code} 时终止。")

        print(f"正在更新 {ts_code} 数据... (剩余超时约 {int(timeout - elapsed_total)}s)")
        attempt = 0
        last_exception = None
        df_new = None

        while attempt < per_code_retries:
            try:
                df_new = pro.fund_daily(ts_code=ts_code)
                # 如果返回为空 DataFrame，视为一次失败并重试
                if df_new is None or df_new.empty:
                    raise ValueError("返回数据为空")
                # 成功则跳出重试循环
                break
            except Exception as e:
                last_exception = e
                attempt += 1
                elapsed_total = time.time() - start_time
                # 如果总超时则立即退出
                if elapsed_total > timeout:
                    raise TimeoutError(f"更新超时（超过 {timeout} 秒），在处理 {ts_code} 时终止。") from e
                if attempt < per_code_retries:
                    print(f"⚠️ {ts_code} 获取失败（第 {attempt}/{per_code_retries} 次），{e}，{retry_delay}s 后重试...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ {ts_code} 达到最大重试次数（{per_code_retries}），跳过。最后错误：{e}")

        # 如果最终 df_new 有效则加入集合
        if df_new is not None and not (df_new is None or (hasattr(df_new, 'empty') and df_new.empty)):
            all_data.append(df_new)
        else:
            # 记录失败但继续处理下一个 code
            print(f"⚠️ 跳过 {ts_code}，未获取到有效数据。最后错误：{last_exception}")

    if not all_data:
        print("未获取到任何新数据，返回本地数据（如果存在）。")
        return df_local

    df_new_all = pd.concat(all_data, ignore_index=True)

    df = pd.concat([df_local, df_new_all]).drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last"
    )
    df = df.sort_values(by=["ts_code", "trade_date"], ascending=[True, True])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"ETF数据已更新并保存到 {csv_path}")

    return df


def plot_heatmap(csv_path="./data/etf.csv", show_days=30, save_path="./output/etf_heatmap.png", save_local=True):
    """
    使用 matplotlib 绘制ETF涨跌幅热力图
    
    参数:
        csv_path: CSV文件路径
        show_days: 展示的交易日数量
        save_path: 图片保存路径
        save_local: 是否保存到本地，默认True
    """
    # 读取数据
    df = pd.read_csv(csv_path, dtype={"ts_code": str, "trade_date": str})
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df[df["ts_code"].isin(flat_codes)]

    # 创建透视表
    pivot_df = df.pivot_table(
        index="trade_date", columns="ts_code", values="pct_chg"
    ).sort_index(ascending=False).head(show_days)

    # 转换代码为ETF名（使用扁平映射）
    ts_to_name = {v: k for k, v in flat_etf.items()}
    pivot_df = pivot_df.rename(columns=ts_to_name)

    # 固定列顺序
    desired_cols = list(flat_etf.keys())
    pivot_df = pivot_df.reindex(columns=desired_cols)

    # 去重
    pivot_df = pivot_df.loc[~pivot_df.index.duplicated(keep='first')]
    pivot_df = pivot_df.loc[:, ~pivot_df.columns.duplicated(keep='first')]

    # ===== 自定义红绿配色（红涨绿跌） =====
    colors = ["#52c41a", "#ffffff", "#ff4d4f"]  # 绿色 -> 白色 -> 红色
    cmap = mcolors.LinearSegmentedColormap.from_list("stock_red_green", colors)

    # 归一化，保证红绿对称
    vmax = abs(pivot_df.max().max())
    vmin = abs(pivot_df.min().min())
    bound = max(vmax, vmin)

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, show_days * 0.3 + 2))
    
    # 绘制热力图
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', 
                   vmin=-bound, vmax=bound)

    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, fontsize=10, fontweight='bold')
    ax.set_yticklabels([d.strftime('%Y-%m-%d') for d in pivot_df.index], fontsize=9)

    # 在每个格子中显示数值
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not pd.isna(value):
                text_color = 'black' if abs(value) < bound * 0.5 else 'white'
                text = ax.text(j, i, f'{value:.2f}%',
                             ha="center", va="center", 
                             color=text_color, fontsize=8, fontweight='bold')

    # 添加标题
    ax.set_title('ETF涨跌幅热力图', fontsize=16, fontweight='bold', pad=15)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('涨跌幅 (%)', rotation=270, labelpad=20, fontsize=11)

    # 调整布局
    plt.tight_layout()

    # 保存图片（可选）
    if save_local:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"热力图已保存到 {save_path}")
    
    plt.close()
    return save_path


def detect_consecutive_negatives(series, min_consecutive=3):
    """
    检测某一列（某个ETF）中连续负值的位置
    
    参数:
        series: pandas Series，某个ETF的收益率序列（按时间降序）
        min_consecutive: 最小连续负值天数，默认3
    
    返回:
        set: 满足连续下跌条件的行索引集合
    """
    marked_indices = set()
    consecutive_count = 0
    consecutive_start = None
    
    for i, (idx, val) in enumerate(series.items()):
        if pd.isna(val):
            # 遇到缺失值，重置计数
            if consecutive_count >= min_consecutive and consecutive_start is not None:
                # 记录之前的连续负值序列
                for j in range(consecutive_start, i):
                    marked_indices.add(series.index[j])
            consecutive_count = 0
            consecutive_start = None
        elif val < 0:
            # 负值，增加计数
            if consecutive_count == 0:
                consecutive_start = i
            consecutive_count += 1
        else:
            # 正值或零，检查之前是否有足够的连续负值
            if consecutive_count >= min_consecutive and consecutive_start is not None:
                for j in range(consecutive_start, i):
                    marked_indices.add(series.index[j])
            consecutive_count = 0
            consecutive_start = None
    
    # 检查最后一段连续负值
    if consecutive_count >= min_consecutive and consecutive_start is not None:
        for j in range(consecutive_start, len(series)):
            marked_indices.add(series.index[j])
    
    return marked_indices


def generate_html_table(csv_path="./data/etf.csv", show_days=30):
    """
    生成 HTML 格式的ETF涨跌幅热力图表格。
    在 ETF 列前增加一列显示该 ETF 所属的分类（字典的键）。
    相邻行分类相同时取消行中间的横线，相同分类的行在类别列进行单元格合并。
    """
    df = pd.read_csv(csv_path, dtype={"ts_code": str, "trade_date": str})
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # 支持扁平化 etf_dict
    df = df[df["ts_code"].isin(flat_codes)]

    # 取涨跌幅透视表（index=trade_date, columns=ts_code）
    pivot_df = df.pivot_table(
        index="trade_date", columns="ts_code", values="pct_chg"
    ).sort_index(ascending=False).head(show_days)

    # 将列名改为 "ETF名称: 代码"（例如 "上证50ETF: 510050.SH"）
    ts_to_label = {v: f"{k}: {v}" for k, v in flat_etf.items()}
    pivot_df = pivot_df.rename(columns=ts_to_label)

    # 按 etf_dict 保持顺序，构建期望的列标签列表（名称: 代码）
    desired_labels = [f"{k}: {v}" for k, v in flat_etf.items()]
    exist_labels = [lbl for lbl in desired_labels if lbl in pivot_df.columns]
    missing_labels = [lbl for lbl in desired_labels if lbl not in pivot_df.columns]
    if missing_labels:
        print(f"⚠️ 以下期望列在数据中未找到（将被忽略）：{missing_labels}")
    if not exist_labels:
        exist_labels = list(pivot_df.columns)
        print("⚠️ 未匹配到预期列，使用数据中现有列显示。")
    pivot_df = pivot_df.reindex(columns=exist_labels)

    # 转置：行 = ETF（"名称: 代码"），列 = 日期（按时间降序）
    pivot_df = pivot_df.T

    # 保证 index 和 columns 唯一
    pivot_df = pivot_df.loc[~pivot_df.index.duplicated(keep='first')]
    pivot_df = pivot_df.loc[:, ~pivot_df.columns.duplicated(keep='first')]

    # 为每行 ETF 构建分类信息（基于 code 提取），并为每个分类分配浅色背景
    category_for_row = {}
    categories = []
    for etf_label in pivot_df.index:
        # etf_label 格式为 "名称: 代码"
        parts = etf_label.rsplit(": ", 1)  # 从右边分割，避免名称中含冒号
        code = parts[1] if len(parts) == 2 else parts[0]
        cat = code_to_group.get(code, "") or "其他"
        category_for_row[etf_label] = cat
        if cat not in categories:
            categories.append(cat)

    # 预定义浅色（pastel）配色，按分类循环分配
    pastel_colors = {
        "国内指数": "#E8F5E9",    # 浅绿
        "国际指数": "#E3F2FD",    # 浅蓝
        "行业题材": "#FFF3E0",    # 浅橙
        "大宗商品": "#F3E5F5",    # 浅紫
        "其他": "#FFFDE7"         # 浅黄
    }
    category_colors = {cat: pastel_colors.get(cat, "#FFFFFF") for cat in categories}

    # ===== 检测每行（每个ETF）中的连续负值 =====
    negative_markers = {}  # {etf_name: set(日期索引)}
    for etf in pivot_df.index:
        negative_markers[etf] = detect_consecutive_negatives(pivot_df.loc[etf], min_consecutive=3)

    # ===== 自定义红绿配色（红涨绿跌） =====
    colors = ["#52c41a", "#ffffff", "#ff4d4f"]
    cmap = mcolors.LinearSegmentedColormap.from_list("stock_red_green", colors)

    vmax = abs(pivot_df.max().max()) if not pivot_df.empty else 0
    vmin = abs(pivot_df.min().min()) if not pivot_df.empty else 0
    bound = max(vmax, vmin)
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound) if bound != 0 else None

    def background_color(val):
        if pd.isna(val) or norm is None:
            return ""
        rgba = cmap(norm(val))
        rgb = tuple(int(x * 255) for x in rgba[:3])
        return f"background-color: rgb{rgb}; color: black;"

    # 为每个日期（每列）找当日涨幅最高的ETF并添加⭐️
    formatted_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns, dtype=object)
    for col in pivot_df.columns:
        col_series = pivot_df[col]
        if col_series.isna().all():
            for etf in pivot_df.index:
                formatted_df.at[etf, col] = ""
            continue
        max_etf = col_series.idxmax()
        for etf in pivot_df.index:
            val = pivot_df.at[etf, col]
            if pd.isna(val):
                formatted_df.at[etf, col] = ""
            elif etf == max_etf:
                formatted_df.at[etf, col] = f"⭐️ {val:.2f}%"
            else:
                formatted_df.at[etf, col] = f"{val:.2f}%"

    # ===== 构建类别合并映射 =====
    # 计算每个类别需要合并的行数和起始位置
    category_merge_info = {}  # {category: [(start_idx, rowspan), ...]}
    etf_list = list(pivot_df.index)
    
    for i, etf in enumerate(etf_list):
        category = category_for_row.get(etf, "")
        if category not in category_merge_info:
            category_merge_info[category] = []
        category_merge_info[category].append(i)
    
    # 合并相邻的行号
    category_spans = {}  # {category: [(start_row, rowspan), ...]}
    for category, row_indices in category_merge_info.items():
        spans = []
        start = row_indices[0]
        rowspan = 1
        
        for i in range(1, len(row_indices)):
            if row_indices[i] == row_indices[i-1] + 1:
                # 连续的行
                rowspan += 1
            else:
                # 不连续，记录当前段并开始新段
                spans.append((start, rowspan))
                start = row_indices[i]
                rowspan = 1
        
        # 记录最后一段
        spans.append((start, rowspan))
        category_spans[category] = spans
    
    # 标记哪些行的类别列已经被输出过（用于避免重复输出）
    category_cell_printed = {}  # {row_idx: True}

    # 手动构建HTML表格（类别列进行合并）
    html_rows = []
    html_rows.append("<table style='border-collapse: collapse; margin: 0 auto; font-size: 14px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.1);'>")
    html_rows.append("<thead><tr style='background-color: #5B8FF9; color: white;'>")
    html_rows.append("<th style='text-align: center; padding: 8px; border: 1px solid #f0f0f0;'>类别</th>")
    html_rows.append("<th style='text-align: center; padding: 8px; border: 1px solid #f0f0f0;'>ETF</th>")
    for col in pivot_df.columns:
        html_rows.append(f"<th style='text-align: center; padding: 8px; border: 1px solid #f0f0f0;'>{col.strftime('%Y-%m-%d')}</th>")
    html_rows.append("</tr></thead><tbody>")

    for row_idx, etf in enumerate(etf_list):
        cur_category = category_for_row.get(etf, "")
        
        # 检查是否需要输出类别列单元格（合并）
        category_cell_html = ""
        if row_idx not in category_cell_printed:
            # 找到该类别对应的合并段信息
            for start, rowspan in category_spans.get(cur_category, []):
                if row_idx == start:
                    # 使用分类对应的浅色背景
                    bg = category_colors.get(cur_category, "#FFFFFF")
                    category_cell_html = f"<td style='text-align: center; padding: 6px 12px; border: 1px solid #d0d0d0; min-width: 100px; font-weight:600; vertical-align: middle; background-color: {bg};' rowspan='{rowspan}'>{cur_category}</td>"
                    # 标记这个段内的所有行为已处理
                    for j in range(start, start + rowspan):
                        category_cell_printed[j] = True
                    break
        
        html_rows.append("<tr>")
        
        # 只在需要时输出类别列单元格
        if category_cell_html:
            html_rows.append(category_cell_html)
        
        # ETF 名称列
        html_rows.append(f"<td style='text-align: left; padding: 6px 12px; border: 1px solid #f0f0f0; min-width: 140px; font-weight:600;'>{etf}</td>")

        for col in pivot_df.columns:
            val = pivot_df.at[etf, col]
            formatted_val = formatted_df.at[etf, col]

            base_style = 'text-align: center; padding: 6px 12px; min-width: 80px;'

            # 检查是否需要添加红色外框（按 ETF 行检测出的日期集合）
            if col in negative_markers.get(etf, set()):
                border_style = 'border: 3px solid #ff0000;'
            else:
                border_style = 'border: 1px solid #f0f0f0;'

            if not pd.isna(val):
                bg_style = background_color(val)
                html_rows.append(f"<td style='{base_style} {border_style} {bg_style}'>{formatted_val}</td>")
            else:
                html_rows.append(f"<td style='{base_style} {border_style}'></td>")
        html_rows.append("</tr>")

    html_rows.append("</tbody></table>")
    html_table = "\n".join(html_rows)

    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: "Microsoft YaHei", Arial, sans-serif;
                background-color: #f9f9f9;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <h2 style="text-align:center; color:#333;">ETF涨跌幅热力图</h2>
        <p style="text-align:center; color:#666; font-size: 12px;">
            注：连续3个或以上交易日下跌的格子标有红色边框
        </p>
        {html_table}
    </body>
    </html>
    """
    return html_template


def send_email(html_content, subject, receivers, sender, password, smtp_server, smtp_port=465, use_ssl=True):
    """
    发送 HTML 邮件，支持 SSL/非SSL，带错误捕获
    """
    try:
        # 构造邮件
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)

        part = MIMEText(html_content, "html", "utf-8")
        msg.attach(part)

        # 建立连接
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()  # 如果服务器支持 TLS

        # 打招呼
        server.ehlo()

        # 登录
        server.login(sender, password)

        # 发送邮件
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()

        print("✅ 邮件发送成功！")

    except smtplib.SMTPAuthenticationError as e:
        print("❌ 邮件认证失败：请检查账号或密码（应用专用密码是否启用？）")
        print(e)
    except smtplib.SMTPConnectError as e:
        print("❌ 无法连接到SMTP服务器：", e)
    except smtplib.SMTPException as e:
        print("❌ 发送邮件失败：", e)
    except Exception as e:
        print("❌ 未知错误：", e)


if __name__ == "__main__":
    # ===== 配置参数 =====

    load_dotenv()
    SENDER = os.getenv("EMAIL_SENDER")
    PASSWORD = os.getenv("EMAIL_PASS")
    RECEIVERS = os.getenv("EMAIL_RECEIVERS").split(",")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))

    # from tushare_token_manager.token_manager import get_valid_token
    # TUSHARE_TOKEN = get_valid_token() 暂时失效
    TUSHARE_TOKEN = '13b3258a9afebe7d2e75cc4bd460e808282a6368101eb52656ee7345'
    # TUSHARE_TOKEN = os.getenv("TS_TOKEN")
    # print("TUSHARE_TOKEN value:", TUSHARE_TOKEN)

    CSV_PATH = "./data/etfindex.csv"
    IMAGE_PATH = "./output/heatmap.png"  # 仅在需要保存本地图片时使用


    # ===== 执行任务 =====
    # 1. 更新数据
    update_etf_data(csv_path=CSV_PATH, token=TUSHARE_TOKEN, timeout=300, per_code_retries=3, retry_delay=5)
    #update_etf_data(CSV_PATH, TUSHARE_TOKEN)
    
    # 2. 生成HTML表格
    html_report = generate_html_table(CSV_PATH, show_days=7)
    
    # 3. 可选：绘制并保存热力图到本地
    # plot_heatmap(CSV_PATH, show_days=30, save_path=IMAGE_PATH, save_local=True)
    
    # 4. 发送邮件
    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"ETF日报 - {today}"
    send_email(html_report, subject, RECEIVERS, SENDER, PASSWORD, SMTP_SERVER, SMTP_PORT)
