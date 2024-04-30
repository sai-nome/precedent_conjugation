import os
import glob
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
import re
import unicodedata
import MeCab
from typing import List, Dict
import openai
import numpy as np

def get_page_count(pdf_path: str) -> int:
    """
    ページ数取得
    Args
        pdf_path (str): ファイルパス
    Return
        page_count (int): ページ数
    """
    # PDFファイルを開く
    with open(pdf_path, 'rb') as file:
        # PDFパーサーとドキュメントオブジェクトを作成
        parser = PDFParser(file)
        document = PDFDocument(parser)
        
        # PDFPage.create_pages()を使用してページのリストを取得し、その長さを数える
        page_count = len(list(PDFPage.create_pages(document)))
        
        return page_count

def remove_whitespaces(text: str, page_count: int) -> str:
    """
    ページ番号の削除
    空白の削除
    Args
        text (str): 抽出したテキスト
        page_count (int): ページ数
    Return
        text (str): 空白削除後のテキスト
    """
    # ページ番号の削除(例：- 1 -)
    for i in range(1, page_count+1):
        text = text.replace('- ' + str(i) + ' -', '')
    # スペース、タブ、改行を削除
    text = text.replace(' ', '').replace('\t', '').replace('\n', '')

    # 正規表現を使って他の空白文字を削除
    text = re.sub(r'\s+', '', text)

    return text

def remove_special_characters(text: str) -> str:
    """
    特殊文字の削除
    Args
        text (str): 抽出したテキスト
    Return
        text (str): 特殊文字削除後のテキスト
    """
    # URLの除去
    text = re.sub(r'http[s]?://\S+', '', text)
    # メールアドレスの除去
    text = re.sub(r'\S*@\S*\s?', '', text)
    # 数学記号などの特殊文字の除去 (必要に応じてパターンを追加)
    text = re.sub(r'[!#\$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)
    return text

def normalize_text(text: str) -> str:
    """
    大文字小文字統一
    Args
        text (str): 抽出したテキスト
    Return
        text (str): 文字統一後のテキスト
    """
    # 全角文字を半角文字に統一（アルファベット、数字、記号）
    normalized_text = ''.join([unicodedata.normalize('NFKC', char) if not unicodedata.east_asian_width(char) in ('F', 'W', 'A') else char for char in text])
    return normalized_text

def split_text_into_sentences(text: str) -> list[str]:
    """
    文末文字で分割リスト化
    Args
        text      (str): 抽出したテキスト
    Return
        sentences (str): 分割後の文リスト
    """
    # 文末記号で分割し、分割された各部分の末尾に元の文末記号を追加してリストに格納
    sentences = re.split(r'([。！？])', text)
    # 分割されたリストを再構成して、文末記号を含む完全な文にする
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
    return sentences

def split_text_with_mecab(text: str) -> list[str]:
    # MeCabの初期化
    mecab = MeCab.Tagger('')

    # テキストを形態素解析
    node = mecab.parseToNode(text)
    
    sentences = []  # 分割された文を格納するリスト
    sentence = ''  # 現在の文
    
    # 形態素ノードをループで処理
    while node:
        # 表層形を現在の文に追加
        sentence = node.surface
        
        if len(sentence) != 0:
            sentences.append(sentence)
        # 品詞情報から句点、読点、助詞を判断し、モデルの性能によっては削除
        # if node.feature.startswith('記号,句点') or node.feature.startswith('記号,読点') or node.feature.startswith('助詞,終助詞'):
        #     pass
        # else:
        #     sentences.append(sentence)
        # if node.feature.startswith('名詞') or node.feature.startswith('動詞') or node.feature.startswith('接頭詞'):
        #     sentences.append(sentence)
        
        node = node.next
    
    # 最後の文が空でなければリストに追加
    if sentence:
        sentences.append(sentence)
    
    return sentences

def split_text2vector(text: str, max_tokens=6000) -> List[float]:
    vectors = []
    for i in range(0, len(text), max_tokens):
        # 単語リストを最大トークン数に基づいて分割
        segment = text[i:i+max_tokens]
        # 分割したテキストでリクエストを送る
        res = openai.embeddings.create(
            model='text-embedding-3-large',
            input=segment
        )
        vectors.append(res.data[0].embedding)
    average_vector = np.mean(vectors, axis=0).tolist()
    return average_vector


def convert_text_to_token_ids(text: str, filename: str, vector_database: Dict[str, str]) -> Dict[str, str]:
    """
    単語IDのリストにパディングを適用し、アテンションマスクを生成します。
    
    :param input_ids: トークン化されたテキストの単語IDのリスト
    :param max_length: パディングを適用する最大の長さ
    :return: パディングが適用された単語IDのリストとアテンションマスク
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    vector = split_text2vector(text)

    # ベクトルをデータベースに追加
    vector_database.append({
        'title': filename,
        'body': text,
        'embedding': vector
    })
    
    return vector_database


def cosine_similarity(v1, v2):
    # ベクトルのドット積
    dot_product = np.dot(v1, v2)
    # ベクトルのノルム
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # コサイン類似度
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

# 現在のディレクトリ取得
dirname = os.getcwd()
# pdf出力用フォルダ
output_pdf = dirname + '/output_pdf'
# 入力ファイル取得
file_path_list = glob.glob(output_pdf+'/*.pdf')
# テスト用カウント
count = 0
# テスト用ベクトルデータベース
vector_database = []

for pdf_path in file_path_list:
    count += 1
    # テスト用pdfファイル
    # pdf_path = 'output_pdf/archive/035867_hanrei.pdf'
    filename = os.path.basename(pdf_path)
    # print(pdf_path)
    # PDFファイルからテキスト抽出
    texts = extract_text(pdf_path)
    # print(texts)
    print('--------------------------------------------------')
    # ページ数取得
    page_count = get_page_count(pdf_path)
    # print(page_count)
    print('--------------------------------------------------')
    # ページ番号と空白削除
    rem_spe_texts = remove_whitespaces(texts, page_count)
    # print(rem_spe_texts)
    print('--------------------------------------------------')
    # 特殊文字削除
    rem_spe_cha_texts = remove_special_characters(rem_spe_texts)
    # print(rem_spe_cha_texts)
    print('--------------------------------------------------')
    # 大文字小文字統一
    nor_texts = normalize_text(rem_spe_cha_texts)
    # print(nor_texts)
    print('--------------------------------------------------')
    # '。','!','?'で分割
    # split_texts = split_text_into_sentences(nor_texts)
    # print(split_texts)
    # print('--------------------------------------------------')
    # MeCabで形態素解析
    # mecab_texts = split_text_with_mecab(nor_texts)
    # print(mecab_texts)
    # print('--------------------------------------------------')
    #  単語IDへの変換
    token_texts = convert_text_to_token_ids(nor_texts, filename, vector_database)
    # print(token_texts)
    print('--------------------------------------------------')    
    if count >= 5:
        break

# これが検索用の文字列
QUERY = '保険金が欲しい'
# 検索用の文字列をベクトル化
query = openai.embeddings.create(
    model='text-embedding-3-large',
    input=QUERY
)

query = query.data[0].embedding

# 総当りで類似度を計算
results = map(
        lambda i: {
            'title': i['title'],
            'body': i['body'],
            # ここでクエリと各文章のコサイン類似度を計算
            'similarity': cosine_similarity(i['embedding'], query)
            },
        vector_database
)
# コサイン類似度で降順（大きい順）にソート
results = sorted(results, key=lambda i: i['similarity'], reverse=True)
# 以下で結果を表示
print(f"Query: {QUERY}")
print("Rank: Title Similarity")
for i, result in enumerate(results):
    print(f'{i+1}: {result["title"]} {result["similarity"]}')

print("====Best Doc====")
print(f'title: {results[0]["title"]}')
print(f'body: {results[0]["body"]}')

print("====Worst Doc====")
print(f'title: {results[-1]["title"]}')
print(f'body: {results[-1]["body"]}')

