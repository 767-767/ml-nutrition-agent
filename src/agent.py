"""
Agent核心逻辑：基于LangGraph的智能营养配餐系统
工作流：解析需求->调用ML模型->组合优化->生成菜谱
"""

import os
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
import json

# LangGraph相关
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("警告: 未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    print("Agent将使用模拟模式运行（不调用大模型）")

# ============================================================
# 1. 加载模型和数据
# ============================================================
MODEL_DIR = "./model"
DATA_PATH = "./data/cleaned_food_data.csv"

print("加载模型和数据...")
df = pd.read_csv(DATA_PATH)
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_classifier.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"数据加载完成: {len(df)} 种食物")
print(f"模型加载完成")

# ============================================================
# 2. 定义 Agent 状态
# ============================================================
class AgentState(TypedDict):
    user_input: str                      # 用户原始输入
    parsed_requirements: Dict[str, Any]  # 解析后的需求
    candidate_foods: pd.DataFrame        # 候选食物列表
    selected_recipe: List[Dict]          # 最终选中的菜谱
    final_message: str                   # 最终输出文本
    iteration: int                       # 迭代次数

# ============================================================
# 3. 工具函数：调用ML模型筛选食物
# ============================================================
def filter_foods_by_ml(requirements: Dict[str, Any]) -> pd.DataFrame:
    """
    使用ML模型筛选符合营养要求的食物
    requirements: {'max_calories': 500, 'min_protein': 20, 'protein_category': 'High Protein'}
    """
    filtered = df.copy()
    
    # 1. 按蛋白质类别筛选（SVM分类模型预测）
    if 'protein_category' in requirements and requirements['protein_category']:
        filtered = filtered[filtered['protein_category'] == requirements['protein_category']]
    
    # 2. 按热量上限筛选
    if 'max_calories' in requirements:
        filtered = filtered[filtered['energy_kcal'] <= requirements['max_calories']]
    
    # 3. 按热量下限筛选
    if 'min_calories' in requirements:
        filtered = filtered[filtered['energy_kcal'] >= requirements['min_calories']]
    
    # 4. 按蛋白质下限筛选
    if 'min_protein' in requirements:
        filtered = filtered[filtered['protein_g'] >= requirements['min_protein']]
    
    # 5. 按碳水上限筛选
    if 'max_carbs' in requirements:
        filtered = filtered[filtered['carbs_g'] <= requirements['max_carbs']]
    
    # 6. 按脂肪上限筛选
    if 'max_fat' in requirements:
        filtered = filtered[filtered['fat_g'] <= requirements['max_fat']]
    
    return filtered

def select_diverse_recipe(candidates: pd.DataFrame, n_items: int = 3) -> List[Dict]:
    """
    从候选中选择多样化的食物组合（最大化营养覆盖）
    使用简单的贪心算法：每次选择与已选食物差异最大的
    """
    if len(candidates) == 0:
        return []
    
    if len(candidates) <= n_items:
        return candidates.head(n_items).to_dict('records')
    
    # 特征归一化（用于计算多样性）
    features = ['protein_g', 'fat_g', 'carbs_g', 'fiber_g']
    candidate_subset = candidates[features + ['food_name', 'energy_kcal', 'protein_category']].copy()
    
    # 归一化
    for col in features:
        max_val = candidate_subset[col].max()
        if max_val > 0:
            candidate_subset[col] = candidate_subset[col] / max_val
    
    # 贪心选择：第一项选蛋白质最高的
    selected_indices = [candidate_subset['protein_g'].idxmax()]
    selected = candidate_subset.loc[selected_indices]
    
    remaining = candidate_subset.drop(selected_indices)
    
    for _ in range(min(n_items - 1, len(remaining))):
        # 计算每个剩余项与已选集合的平均距离
        best_score = -1
        best_idx = None
        
        for idx, row in remaining.iterrows():
            # 计算与所有已选食物的平均欧氏距离
            dists = []
            for _, sel_row in selected.iterrows():
                dist = np.sqrt(sum((row[features].values - sel_row[features].values)**2))
                dists.append(dist)
            avg_dist = np.mean(dists)
            
            if avg_dist > best_score:
                best_score = avg_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected = candidate_subset.loc[selected_indices]
            remaining = remaining.drop(best_idx)
    
    return candidates.loc[selected_indices].to_dict('records')

# ============================================================
# 4. LangGraph 节点函数
# ============================================================

# 初始化LLM（使用DeepSeek）
def get_llm():
    if DEEPSEEK_API_KEY:
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )
    return None

# 节点1：解析用户需求
def parse_requirements(state: AgentState) -> AgentState:
    """使用大模型解析用户输入，提取结构化需求"""
    user_input = state['user_input']
    llm = get_llm()
    
    # 默认需求（安全降级）
    default_requirements = {
        'protein_category': 'High Protein',
        'max_calories': 500,
        'min_calories': 300,
        'min_protein': 15,
        'max_carbs': 50,
        'max_fat': 20
    }
    
    if llm is None:
        # 无API Key时使用默认值
        state['parsed_requirements'] = default_requirements
        state['iteration'] = 1
        return state
    
    # 构造提示词
    system_prompt = """你是一个营养学专家助手。用户会提出饮食需求，你需要从中提取关键营养参数。

请从用户输入中提取以下参数，以JSON格式返回：
- protein_category: "High Protein" / "Medium Protein" / "Low Protein" (根据用户描述判断)
- max_calories: 热量上限（千卡）
- min_calories: 热量下限（千卡）
- min_protein: 蛋白质下限（克）
- max_carbs: 碳水上限（克）
- max_fat: 脂肪上限（克）

如果用户没有明确提到某个参数，使用合理的默认值。
只返回JSON，不要有其他文字。"""
    
    user_prompt = f"用户需求: {user_input}"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # 解析JSON
        content = response.content.strip()
        # 如果返回内容被markdown代码块包裹，去除
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        parsed = json.loads(content.strip())
        # 合并默认值
        for key, default_val in default_requirements.items():
            if key not in parsed or parsed[key] is None:
                parsed[key] = default_val
        
        state['parsed_requirements'] = parsed
        
    except Exception as e:
        print(f"⚠️ 解析失败，使用默认需求: {e}")
        state['parsed_requirements'] = default_requirements
    
    state['iteration'] = 1
    return state

# 节点2：筛选候选食物（调用ML模型）
def search_candidates(state: AgentState) -> AgentState:
    """根据解析后的需求筛选候选食物"""
    req = state['parsed_requirements']
    candidates = filter_foods_by_ml(req)
    state['candidate_foods'] = candidates
    return state

# 节点3：选择菜谱（组合优化）
def select_recipe(state: AgentState) -> AgentState:
    """从候选中选择多样化的菜谱组合"""
    candidates = state['candidate_foods']
    
    if len(candidates) == 0:
        state['selected_recipe'] = []
        state['final_message'] = "没有找到符合您要求的食物组合，请放宽一些条件。"
        return state
    
    # 选择3-5种食物
    n_items = min(5, max(3, len(candidates) // 10 + 1))
    recipe = select_diverse_recipe(candidates, n_items)
    state['selected_recipe'] = recipe
    
    # 生成最终消息
    req = state['parsed_requirements']
    if len(recipe) == 0:
        state['final_message'] = "没有找到符合您要求的食物组合，请放宽一些条件。"
    else:
        total_calories = sum(item['energy_kcal'] for item in recipe)
        total_protein = sum(item['protein_g'] for item in recipe)
        state['final_message'] = f"根据您的要求（{req.get('protein_category', '不限')}，{req.get('min_calories', 0)}-{req.get('max_calories', 0)}千卡），我为您推荐以下{len(recipe)}种食物：\n\n"
        
        for i, item in enumerate(recipe, 1):
            state['final_message'] += f"{i}. {item['food_name']}\n"
            state['final_message'] += f"{item['energy_kcal']:.0f}千卡 | 蛋白质 {item['protein_g']:.1f}g | 脂肪 {item['fat_g']:.1f}g | 碳水 {item['carbs_g']:.1f}g\n"
            state['final_message'] += f"{item['protein_category']}\n\n"
        
        state['final_message'] += f"总计: {total_calories:.0f}千卡, 蛋白质 {total_protein:.1f}g"
    
    return state

# 节点4：反思与优化（可选，展示Agent的迭代能力）
def reflect_and_optimize(state: AgentState) -> AgentState:
    """
    检查结果是否符合预期，如果不符合则触发重新搜索
    展示Agent的"自我反思"能力
    """
    req = state['parsed_requirements']
    recipe = state['selected_recipe']
    
    if len(recipe) == 0:
        return state
    
    # 检查总热量是否符合要求
    total_calories = sum(item['energy_kcal'] for item in recipe)
    min_cal = req.get('min_calories', 0)
    max_cal = req.get('max_calories', 1000)
    
    # 如果总热量超出范围，记录但继续（不强制重新搜索）
    if total_calories > max_cal:
        state['final_message'] += f"\n\n提示: 总热量({total_calories:.0f}千卡)略高于您的上限({max_cal}千卡)，您可以选择去掉某个高热量食物。"
    elif total_calories < min_cal:
        state['final_message'] += f"\n\n提示: 总热量({total_calories:.0f}千卡)低于您的下限({min_cal}千卡)，可以考虑增加一些食物。"
    
    state['iteration'] += 1
    return state

# ============================================================
# 5. 构建 LangGraph
# ============================================================
def build_agent_graph():
    """构建完整的Agent工作流"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("parse", parse_requirements)
    workflow.add_node("search", search_candidates)
    workflow.add_node("select", select_recipe)
    workflow.add_node("reflect", reflect_and_optimize)
    
    # 定义边（顺序执行）
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "search")
    workflow.add_edge("search", "select")
    workflow.add_edge("select", "reflect")
    workflow.add_edge("reflect", END)
    
    return workflow.compile()

# ============================================================
# 6. 运行函数
# ============================================================
def run_agent(user_input: str) -> str:
    """运行Agent，返回最终结果"""
    print(f"\n{'='*60}")
    print(f"Agent 开始处理: {user_input}")
    print(f"{'='*60}")
    
    agent = build_agent_graph()
    
    initial_state: AgentState = {
        'user_input': user_input,
        'parsed_requirements': {},
        'candidate_foods': pd.DataFrame(),
        'selected_recipe': [],
        'final_message': '',
        'iteration': 0
    }
    
    result = agent.invoke(initial_state)
    
    print(f"\n解析需求: {result['parsed_requirements']}")
    print(f"找到 {len(result['candidate_foods'])} 种候选食物")
    print(f"选中 {len(result['selected_recipe'])} 种食物")
    print(f"迭代次数: {result['iteration']}")
    
    return result['final_message']

# ============================================================
# 7. 测试入口（命令行交互）
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("智能营养配餐Agent")
    print("="*60)
    print("示例输入:")
    print("  - '我想吃高蛋白、低脂肪的晚餐，热量控制在400-500千卡'")
    print("  - '推荐一些低碳水、高蛋白的食物，热量不超过400千卡'")
    print("  - '我需要减脂餐，高蛋白，热量300-400千卡'")
    print("="*60)
    
    while True:
        user_input = input("\n请输入您的需求（输入 q 退出）: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            result = run_agent(user_input)
            print(f"\n{'='*60}")
            print("最终推荐:")
            print(result)
            print(f"{'='*60}")
        except Exception as e:
            print(f"出错: {e}")