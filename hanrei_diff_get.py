from bs4 import BeautifulSoup
import urllib.request as req
import urllib
import os
import time
import glob
from http.client import RemoteDisconnected

# 現在のディレクトリ取得
dirname = os.getcwd()
# pdf出力用フォルダ
output_pdf = dirname + '/output_pdf'
# ページ数をカウント(処理を中断した箇所から再開)→2024/05/10時点で6550page
page = 0
# 基本URL
root_url = 'https://www.courts.go.jp/'

# 繰り返し処理
while True:
# for _ in range(1): # テストコード
    # ページ数
    pdf_count = 0
    page += 1
    print('page数: ',page)
    # 全件検索結果URL
    url = root_url + 'app/hanrei_jp/list1?page='+str(page)+'&sort=1&filter%5BjudgeDateMode%5D=2&filter%5BjudgeGengoFrom%5D=%E6%98%AD%E5%92%8C&filter%5BjudgeYearFrom%5D=1'
    try:
        res = req.urlopen(url)
        soup = BeautifulSoup(res, 'html.parser')
        time.sleep(2)
    except RemoteDisconnected:
        print('RemoteDisconnectedのページ数: ',page)
        continue
    result = soup.select('a[href]')
    # リンクを取得したものをループで処理
    for link in result:
        count = 0
        href = link.get('href')
        # pdfのリンクがあれば処理を実行
        if href.endswith('.pdf'):
            pdf_count += 1
            # pdf用のURL作成
            pdf_url = root_url + href
            # 保存用ファイル名取得
            file_name = href.split('/')[len(href.split('/'))-1]
            # 保存用ファイルパス作成
            pdf_path = os.path.join(output_pdf, file_name)
            
            # ファイル名リストを作って取り出して同じものがあるかのチェックを入れる
            # 出力用フォルダからファイルパスを抽出
            file_path_list = glob.glob(output_pdf+'/*')
            file_name_list = []
            # ファイルパスからファイル名を抽出しリスト化
            for file_path in file_path_list:
                file_name_list.append(os.path.basename(file_path))
            # 対象のpdfが出力用フォルダに存在しない場合、ダウンロードを行う
            if file_name not in file_name_list:
                for i in range(6,1):
                    # 対象のpdfをダウンロードし、保存
                    try:
                        count += 1
                        urllib.request.urlretrieve(pdf_url, pdf_path)
                        print(pdf_url)
                        # 連続アクセス防止
                        time.sleep(3)
                        break
                    except Exception as e:
                        if count <= 5:
                            print("5回まで再処理を実施します: ", count)
                            continue
                        else:
                            raise e 
            # sys.exit(1)
    # pdfが存在しなかった場合、処理を終了する
    if (pdf_count == 0):
        print('判例PDFがありません')
        break