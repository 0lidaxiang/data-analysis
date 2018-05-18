
from bs4 import BeautifulSoup
import requests

url = 'http://finance.sina.com.cn/stock/'
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import json
import re

# url = 'http://news.sina.com.cn/c/nd/2017-05-08/doc-ifyeycfp9368908.shtml'
web_data = requests.get(url)
web_data.encoding = 'utf-8'
soup = BeautifulSoup(web_data.text,'lxml')
print(soup)
