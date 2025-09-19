import requests
import json


def seaxng_search(query):
    """输入搜索内容，使用SearXNG进行搜索"""

    SEARXNG_URL = "http://127.0.0.1:6688"
    SEARXNG_ENFING_TOKEN = '123456789'
    params = {}

    # 设置参数
    params['q'] = query
    params['format'] = 'json'
    params['token'] = SEARXNG_ENFING_TOKEN

    # 发送get 请求
    response = requests.get(SEARXNG_URL, params=params)
    #return response.text
    # 检查相应状态码。
    if response.status_code == 200:
        res = response.json()
        relist = []
        for item in res['results']:
            relist.append({
                'title':item['title'],
                'url':item['url'],
                'content':item['content'],
            })
            if len(relist) >= 3:
                break
        return relist
    else:
        response.raise_for_status()

if __name__ == '__main__':
    print(seaxng_search('python'))