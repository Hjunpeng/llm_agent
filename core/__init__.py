import os
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")



if __name__ == '__main__':
    print(DEEPSEEK_API_KEY)