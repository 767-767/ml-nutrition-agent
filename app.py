"""
Gradio Web 界面
"""

import gradio as gr
from src.agent import run_agent

def chat(message, history):
    """
    处理用户消息，返回Agent响应
    message: 当前用户输入
    history: 对话历史（Gradio自动维护）
    """
    if not message or not message.strip():
        return "请输入有效的问题"
    
    result = run_agent(message.strip())
    return result

# 创建界面
demo = gr.ChatInterface(
    fn=chat,
    title="智能营养配餐系统",
    description="输入您的饮食需求，我会推荐合适的食物组合。",
    examples=[
        "我想吃高蛋白、低脂肪的晚餐，热量控制在400-500千卡",
        "推荐一些低碳水、高蛋白的食物，热量不超过400千卡",
        "我需要减脂餐，高蛋白，热量300-400千卡"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)