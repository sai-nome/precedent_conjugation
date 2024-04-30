from bs4 import BeautifulSoup
import urllib.request as req
import urllib
import os
import time

# 現在のディレクトリ取得
dirname = os.getcwd()
# pdf出力用フォルダ
output_pdf = dirname + "/output_pdf"
# ページ数やpdf数をカウント
page = pdf_count = 0
# 基本URL
root_url = "https://www.courts.go.jp/"

# 繰り返し処理
while True:
# for _ in range(1): # テストコード
    # ページ数
    pdf_count = 0
    page += 1
    # 全件検索結果URL
    url = root_url + "app/hanrei_jp/list1?page="+str(page)+"&sort=1&filter%5BjudgeDateMode%5D=2&filter%5BjudgeGengoFrom%5D=%E6%98%AD%E5%92%8C&filter%5BjudgeYearFrom%5D=1"
    res = req.urlopen(url)
    soup = BeautifulSoup(res, "html.parser")
    result = soup.select("a[href]")
    # リンクを取得したものをループで処理
    for link in result:
        href = link.get("href")
        # pdfのリンクがあれば処理を実行
        if href.endswith('.pdf'):
            # pdf用のURL作成
            pdf_url = root_url + href
            # 保存用ファイル名取得
            file_name = href.split("/")[len(href.split("/"))-1]
            # 保存用ファイルパス作成
            pdf_path = os.path.join(output_pdf, file_name)
            
            # 対象のpdfをダウンロードし、保存
            urllib.request.urlretrieve(pdf_url, pdf_path)
            # 連続アクセス防止
            time.sleep(3)
            pdf_count = 1
    # pdfが存在しなかった場合、処理を終了する
    if pdf_count == 0:
        print("判例PDFがありません")
        break