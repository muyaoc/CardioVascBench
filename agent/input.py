import requests
import time
import os
import sys

# 这个脚本用于“批量把本地 Markdown 文档提交到 Medical Graph RAG API 进行入库”。
#
# 工作流程：
# - 扫描指定目录下的所有 .md 文件（递归）
# - 逐个读取文件内容，并调用后端接口 POST /api/admin/ingest 提交入库任务
# - 提交后轮询 GET /api/admin/progress 展示服务端处理进度（带断线重连/重试）
#
# 适用场景：
# - 你已经启动了后端服务（默认 http://localhost:8801）
# - 你希望把 clean_data 目录中的 markdown 文本批量写入 Neo4j（Chunk + 向量索引 + 图谱抽取）

# ================= 配置部分 =================
# 后端服务地址
BASE_URL = "http://localhost:8801"

# ✅ 鉴权配置（与服务端 config.py 中的 token 保持一致）
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-drug",
}

# 🔴 请修改为你存放 Markdown 文件的【目录路径】（会递归扫描其所有子目录）
SOURCE_DIRECTORY = "/data/RAG/clean_data2"

# ================= 终端颜色（ANSI 转义序列） =================
GREEN = "\033[92m"   # 成功/进度条高亮
RED = "\033[91m"     # 错误
CYAN = "\033[96m"    # 提示/监控信息
YELLOW = "\033[93m"  # 警告/重试
RESET = "\033[0m"    # 重置颜色
BOLD = "\033[1m"     # 加粗
GREY = "\033[90m"    # 置灰（用于显示总体进度）

def get_all_md_files(directory):
    """递归查找目录下所有 .md 文件。

    参数:
        directory: 要扫描的根目录

    返回:
        按路径排序后的 Markdown 文件绝对/相对路径列表（取决于 directory 传入形式）
    """
    md_files = []

    # 目录不存在时直接返回空列表，由上层统一提示
    if not os.path.exists(directory):
        return []

    # os.walk 会递归遍历所有子目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".md"):
                md_files.append(os.path.join(root, file))

    return sorted(md_files)

def monitor_progress(filename):
    """
    监控当前文件的处理进度 (增强版：支持断线重连)
    返回: True (成功完成), False (失败/报错)
    """
    print(f"{CYAN}   📡 正在监控服务端处理进度...{RESET}")
    bar_length = 30
    retry_count = 0
    max_retries = 10  # 允许监控连接失败 10 次
    
    while True:
        try:
            # 增加 timeout 防止网络卡死
            res = requests.get(f"{BASE_URL}/api/admin/progress", headers=HEADERS, timeout=10)
            
            # 如果是 200，重置重试计数
            if res.status_code == 200:
                retry_count = 0
                status = res.json()
                is_running = status.get("is_running", False)
                total = status.get("total_batches", 0)
                current = status.get("current_batch", 0)
                error = status.get("error")
                total_docs = status.get("total_docs", 0)

                # 1. 检查报错
                if error:
                    print(f"\n{RED}   ❌ 服务器返回错误: {error}{RESET}")
                    # 服务器报错了，但我们要给它一点时间恢复，或者直接判定失败
                    return False

                # 2. 计算进度条
                if total > 0:
                    percent = min(100, int((current / total) * 100))
                    filled = int(bar_length * current // total)
                    bar = '█' * filled + '-' * (bar_length - filled)
                else:
                    percent = 0
                    bar = '-' * bar_length

                # 3. 打印动态刷新
                sys.stdout.write(f"\r   [{GREEN}{bar}{RESET}] {percent}% | 批次: {current}/{total} | 片段: {total_docs}   ")
                sys.stdout.flush()

                # 4. 判定完成
                if not is_running:
                    if total > 0 and current >= total:
                        print(f"\n{GREEN}   ✅ [{filename}] 处理完成！{RESET}")
                        return True
                    elif total == 0:
                        # 任务还没开始，继续等
                        pass
                    else:
                        # 任务停止了，但没跑完
                        # 可能是之前的任务超时了，我们给它一个机会
                        print(f"\n{YELLOW}   ⚠️ 任务似乎意外停止 ({current}/{total})。尝试重新提交...{RESET}")
                        return False # 返回 False 让主循环决定是否重试
            else:
                # 状态码不是 200，可能是服务器重启或过载
                print(f"\r{YELLOW}   ⚠️ 监控连接异常 ({res.status_code})，重试中...{RESET}", end="")
                retry_count += 1

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            retry_count += 1
            print(f"\r{YELLOW}   ⚠️ 网络超时/连接中断，正在重连 ({retry_count}/{max_retries})...{RESET}", end="")
        
        # 超过重试次数，判定失败
        if retry_count >= max_retries:
            print(f"\n{RED}   ❌ 监控连接彻底丢失，跳过此文件。{RESET}")
            return False

        time.sleep(2)

def process_single_file(filepath):
    """处理单个文件的上传和等待"""
    filename = os.path.basename(filepath)
    print(f"\n{BOLD}📄 正在处理文件: {filename}{RESET}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        print(f"   大小: {size_mb:.2f} MB | 路径: {filepath}")

        # 构造请求
        payload = {
            "text": content,
            "source": filename.replace(".md", ""),
            "agent_name": "batch_uploader"
        }

        # 循环尝试提交 (防止上一个任务还没彻底释放锁)
        max_submit_retries = 5
        for i in range(max_submit_retries):
            try:
                res = requests.post(f"{BASE_URL}/api/admin/ingest", json=payload, headers=HEADERS, timeout=10)
                
                if res.status_code == 200:
                    print(f"{GREEN}   ✔ 上传成功，等待后台处理...{RESET}")
                    time.sleep(3) # 给服务器一点反应时间
                    
                    # 开始监控，如果监控返回 True 代表成功，False 代表失败
                    if monitor_progress(filename):
                        return True
                    else:
                        # 如果监控返回失败（比如任务中途断了），我们可以选择重试上传
                        print(f"{YELLOW}   🔄 任务中断，尝试重新上传 ({i+1}/{max_submit_retries})...{RESET}")
                        time.sleep(5)
                        continue
                
                elif res.status_code == 400:
                    # 400 通常意味着 Job already running
                    print(f"\r{YELLOW}   ⏳ 服务器忙 (上个任务收尾中)，等待 5秒后重试...{RESET}", end="")
                    time.sleep(5)
                else:
                    print(f"{RED}   ❌ 上传请求失败: {res.status_code} - {res.text}{RESET}")
                    time.sleep(5)
            
            except requests.exceptions.RequestException as e:
                print(f"{RED}   ❌ 请求异常: {e}{RESET}")
                time.sleep(5)
        
        print(f"\n{RED}   ❌ 多次尝试失败，跳过文件: {filename}{RESET}")
        return False

    except Exception as e:
        print(f"{RED}   ❌ 读取文件异常: {e}{RESET}")
        return False

def main():
    print(f"{GREEN}=== 📦 批量数据入库脚本 (增强版) ==={RESET}")
    print(f"扫描目录: {SOURCE_DIRECTORY}")
    
    files = get_all_md_files(SOURCE_DIRECTORY)
    
    if not files:
        print(f"{RED}❌ 未找到任何 .md 文件。请检查路径配置。{RESET}")
        return

    print(f"共发现 {BOLD}{len(files)}{RESET} 个 Markdown 文件，准备开始处理...\n")
    print("-" * 50)

    success_count = 0
    fail_count = 0
    failed_files = []

    try:
        for idx, filepath in enumerate(files):
            print(f"{GREY}进度: {idx+1}/{len(files)}{RESET}")
            
            if process_single_file(filepath):
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(os.path.basename(filepath))
            
            print("-" * 50)
            # 文件之间缓冲
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}🛑 用户手动停止脚本。{RESET}")

    # === 最终汇总 ===
    print(f"\n{BOLD}=== 🎉 批量处理结束 ==={RESET}")
    print(f"总计文件: {len(files)}")
    print(f"{GREEN}成功入库: {success_count}{RESET}")
    
    if fail_count > 0:
        print(f"{RED}处理失败: {fail_count}{RESET}")
        print("失败文件列表:")
        for f in failed_files:
            print(f"- {f}")
        print(f"\n{YELLOW}💡 提示: 您可以重新运行此脚本，它会自动尝试重传失败的文件。{RESET}")
    else:
        print(f"{GREEN}完美！所有文件均已处理。{RESET}")

if __name__ == "__main__":
    main()