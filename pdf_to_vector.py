import os
import glob
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
import re
import unicodedata
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import shutil

class PDFToVector:
    def __init__(self, model_name: str):
        # モデルを読み込み
        self.model = SentenceTransformer(model_name)

    def get_page_count(self, pdf_path: str) -> int:
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

    def remove_whitespaces(self, text: str, page_count: int) -> str:
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

    def remove_special_characters(self, text: str) -> str:
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

    def normalize_text(self, text: str) -> str:
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

    def split_text2vector(self, text: str, max_tokens=200) -> List[float]:
        """
        文章をベクトルに変換する関数
        Args:
            text (str): ベクトル化する文章
        Returns:
            list: ベクトル化された文章
        """
        vectors = []
        for i in range(0, len(text), max_tokens):
            # 単語リストを最大トークン数に基づいて分割
            segment = text[i:i+max_tokens]
            # 分割したテキストでリクエストを送る
            vector = self.model.encode(segment, convert_to_tensor=True)
            vectors.append(vector)
        average_vector = np.mean(vectors, axis=0).tolist()
        return average_vector

    def embed_sentence(self, text: str, vector_fol: str, filename: str, pdf_path: str, archive_pdf: str):
        """
        文章をベクトルに変換する関数
        Args:
            text (str): ベクトル化する文章
        Returns:
            list: ベクトル化された文章
        """
        # テキストを分割してベクトル化
        embedding = self.split_text2vector(text)

        # ファイル別に保存
        vector = {
            'title': filename,
            'body': text,
            'embedding': embedding
        }
        vector_file = vector_fol + "/" + filename + ".json"
        with open(vector_file, 'w', encoding='utf-8') as file:
            json.dump(vector, file, ensure_ascii=False, indent=4)

        # アーカイブ化
        shutil.move(pdf_path, archive_pdf)

    def main(self):
        # 現在のディレクトリ取得
        dirname = os.getcwd()
        # pdf出力用フォルダ
        output_pdf = dirname + '/output_pdf'
        # 入力ファイル取得
        file_path_list = glob.glob(output_pdf+'/*.pdf')
        # pdfアーカイブフォルダ
        archive_pdf = output_pdf + '/archive'
        # テスト用ベクトルデータベース
        vector_database = []
        # モデルファイル
        # saved_model = dirname + '/models/proto_model'
        # ベクトルフォルダ
        vector_fol = dirname + '/vector_database'
        # ベクトルデータベース
        vector_db_json = vector_fol + '/vector_database.json'
        # データベース取得
        db_path_list = glob.glob(vector_fol+'/*.json')

        for pdf_path in file_path_list:
            # テスト用pdfファイル
            # pdf_path = 'output_pdf/archive/035867_hanrei.pdf'
            filename = os.path.basename(pdf_path)
            # PDFファイルからテキスト抽出
            texts = extract_text(pdf_path)
            # ページ数取得
            page_count = self.get_page_count(pdf_path)
            # ページ番号と空白削除
            rem_spe_texts = self.remove_whitespaces(texts, page_count)
            # 特殊文字削除
            rem_spe_cha_texts = self.remove_special_characters(rem_spe_texts)
            # 大文字小文字統一
            nor_texts = self.normalize_text(rem_spe_cha_texts)
            # 文章をベクトル変換
            self.embed_sentence(nor_texts, vector_fol, filename, pdf_path, archive_pdf)
            print('filename: ', filename)    

        for db_path in db_path_list:
            json_open = open(db_path, 'r')
            json_load = json.load(json_open)
            vector_database.append(json_load)

        # ベクトルデータベースを格納
        with open(vector_db_json, 'w', encoding='utf-8') as file:
            json.dump(vector_database, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pdf_to_vector = PDFToVector(model_name='paraphrase-multilingual-mpnet-base-v2')
    pdf_to_vector.main()
