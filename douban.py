import requests
from bs4 import BeautifulSoup
import openpyxl

url = 'https://book.douban.com/top250'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

print(response)
print(response.text)

# 创建一个新的Excel文件和工作表
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = 'Douban Books'

# 添加表头
sheet['A1'] = 'Title'
sheet['B1'] = 'Author'
sheet['C1'] = 'Rating'
sheet['D1'] = 'Review Count'
sheet['E1'] = 'Category'  # 添加分类信息

# 获取书籍信息并写入Excel
books = soup.find_all('tr', class_='item')
row = 2  # 从第二行开始写入数据
for book in books:
    title = book.find('td', class_='title').a['title']
    author = book.find('p', class_='pl').text.strip().replace('\n', '')
    rating = book.find('span', class_='rating_num').text
    review_count = book.find('span', class_='pl').text.strip('()')

    # 获取分类信息，如果不存在则设置为空字符串
    category_info = book.find('span', class_='subject-cast')
    category = category_info.text.strip() if category_info else ''

    sheet[f'A{row}'] = title
    sheet[f'B{row}'] = author
    sheet[f'C{row}'] = rating
    sheet[f'D{row}'] = review_count
    sheet[f'E{row}'] = category

    row += 1

# 保存Excel文件
workbook.save('douban_books_with_info_and_category.xlsx')
