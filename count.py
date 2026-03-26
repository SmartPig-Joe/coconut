import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_save_steps_frequency_histogram(file_path, output_filename="steps_frequency_histogram.png"):
    """
    读取JSON文件，统计每个问题中 "steps" 数量的频率，并将频率直方图保存为图片文件。

    Args:
        file_path (str): JSON文件的路径。
        output_filename (str): 输出图片文件的名称。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是一个有效的JSON文件。")
        return

    # 提取每个问题的步骤数
    step_counts = [len(item.get('steps', [])) for item in data]

    if not step_counts:
        print("文件中没有找到任何问题或步骤数据。")
        return

    # 统计不同步骤数的原始次数
    frequency_counts = Counter(step_counts)
    total_questions = len(step_counts)

    # 打印频率统计结果
    print("问题步骤数频率统计：")
    for steps, count in sorted(frequency_counts.items()):
        # 计算频率（百分比）
        percentage = (count / total_questions) * 100
        print(f"  包含 {steps} 个步骤的问题频率: {percentage:.2f}% ({count} / {total_questions})")

    # --- 生成并保存直方图 ---
    plt.figure(figsize=(10, 6))
    
    # 使用 density=True 来绘制频率密度直方图
    # 这意味着图的总面积将为1
    weights = np.ones_like(step_counts) / len(step_counts)
    plt.hist(step_counts, 
             bins=range(min(step_counts), max(step_counts) + 2), 
             align='left', 
             rwidth=0.8, 
             weights=weights) # 使用权重来显示频率

    plt.xlabel("步骤数量 (Steps)")
    plt.ylabel("频率 (Frequency)")
    plt.title("问题中步骤数量的频率直方图")
    plt.xticks(range(min(step_counts), max(step_counts) + 1))
    
    # 将Y轴格式化为百分比
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.grid(axis='y', alpha=0.75)
    
    # 保存图表
    plt.savefig(output_filename)
    print(f"\n频率直方图已成功保存为 '{output_filename}'")
    
    # 关闭图形，释放内存
    plt.close()

analyze_and_save_steps_frequency_histogram('/home/coconut/data/gsm_test.json')
