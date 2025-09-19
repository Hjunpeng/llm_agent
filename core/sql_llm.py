import os
import re
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains.sql_database.prompt import PROMPT as SQL_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine


class SqlLLm():
    def __init__(self, llm):
        self.llm = llm
        self.db = self.init_db()

    def init_db(self):
        # 初始化ClickHouse连接
        engine = create_engine(
            f"clickhouse://{os.getenv('CLICKHOUSE_USER')}:{os.getenv('CLICKHOUSE_PASSWORD')}"
            f"@{os.getenv('CLICKHOUSE_HOST')}:8123/{os.getenv('CLICKHOUSE_DB')}"
        )
        db = SQLDatabase(
            engine,
            view_support=True,
        )
        context = db.get_context()
        print(list(context))
        print(2222, context["table_info"])
        return db

    def prompt(self):
        # # 自定义提示模板优化SQL生成
        sql_prompt = SQL_PROMPT.partial(
            dialect="ClickHouse",
            top_k=10,
            # table_info=""""""
        )
        return sql_prompt

    def chain(self, sql_prompt):

        # 创建SQL查询链
        sql_chain = create_sql_query_chain(self.llm, self.db, prompt=sql_prompt)
        return sql_chain

    def extract_sql(self, text):
        """
        从文本中提取SQL语句
        :param text: 包含SQL的文本
        :return: 提取的SQL字符串，如果未找到则返回None
        """
        # 正则表达式匹配SQL代码块
        pattern = r"```sql(.*?)```"

        # 使用非贪婪匹配查找第一个SQL代码块
        match = re.search(pattern, text, re.DOTALL)

        if match:
            # 提取并清理SQL语句
            sql = match.group(1).strip()
            # 移除可能的注释行
            sql = re.sub(r'^--.*$', '', sql, flags=re.MULTILINE)
            return sql.strip()

        return None
    # 增强版SQL处理链（包含结果解释）

    def enhanced_sql_chain(self, question: str):
        # 生成SQL查询
        sql_chain = self.chain(sql_prompt=self.prompt())
        sql_query = sql_chain.invoke({"question": question})
        sql_query = self.extract_sql(sql_query)
        print(2222222, sql_query)
        # 执行查询
        try:
            result = self.db.run(sql_query)
            # 将结果转换为自然语言
            explanation_prompt = ChatPromptTemplate.from_template(
                "用户问题: {question}\n"
                "生成的SQL: {sql_query}\n"
                "查询结果: {result}\n\n"
                "请用自然语言解释结果:"
            )
            explanation_chain = explanation_prompt | self.llm
            return explanation_chain.invoke({
                "question": question,
                "sql_query": sql_query,
                "result": result
            })
        except Exception as e:
            return f"查询执行失败: {str(e)}\n生成的SQL: {sql_query}"


if __name__ == '__main__':

    from core.deepseek_llm import deepseek_llm
    llm = SqlLLm(deepseek_llm)
    response = llm.enhanced_sql_chain("查询近三个月价格在20-50万的SUV每月的销量, 使用表格列出来结果")
    print(response)
