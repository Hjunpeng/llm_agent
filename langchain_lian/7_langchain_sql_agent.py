from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# from langchain.chains import create_sql_query_chain
#
#
# chain = create_sql_query_chain(SQL_PROMPT, engine)