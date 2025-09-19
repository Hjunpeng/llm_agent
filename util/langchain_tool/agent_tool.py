
from langchain.agents import tool
from my_search import seaxng_search

@tool
def get_word_length(word_length):
    """获取字符串长短"""
    return len(word_length)


@tool
def seaxng_search_tool(text):
    """输入搜索内容，使用SearXNG进行搜索"""
    return seaxng_search(text)



if __name__ == '__main__':
    print(get_word_length.invoke('ss'))