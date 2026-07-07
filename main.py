from src.agent import run_agent

if __name__ == "__main__":
    while True:
        user_input = input("\n请输入需求（输入 q 退出）: ").strip()
        if user_input.lower() == 'q':
            break
        if user_input:
            print(run_agent(user_input))