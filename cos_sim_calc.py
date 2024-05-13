import os
import re
import unicodedata
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class CosSimCalc:
    def __init__(self, model_name: str):
        # モデルを読み込み
        self.model = SentenceTransformer(model_name)

    def remove_whitespaces(self, text: str) -> str:
        """
        空白の削除
        Args
            text (str): 抽出したテキスト
            page_count (int): ページ数
        Return
            text (str): 空白削除後のテキスト
        """
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

    def split_text_into_sentences(self, text: str) -> list[str]:
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

    def embed_sentence(self, text: str) -> List:
        """
        文章をベクトルに変換する関数
        Args:
            text (str): ベクトル化する文章
        Returns:
            list: ベクトル化された文章
        """
        # テキストを分割してベクトル化
        embedding = self.split_text2vector(text)
        return embedding

    def cosine_similarity(self, v1, v2):
        # ベクトルのドット積
        dot_product = np.dot(v1, v2)
        # ベクトルのノルム
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # コサイン類似度
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity

    def main(self, question):
        # 現在のディレクトリ取得
        dirname = os.getcwd()
        # ベクトルフォルダ
        vector_fol = dirname + '/vector_database'
        # ベクトルデータベース
        vector_db_json = vector_fol + '/vector_database.json'
        with open(vector_db_json, 'r', encoding="utf-8") as file:
            vector_database = json.load(file)

        # これが検索用の文字列
        # question = '損害賠償について'
        # ページ番号と空白削除
        rem_spe_texts = self.remove_whitespaces(question)
        # 特殊文字削除
        rem_spe_cha_texts = self.remove_special_characters(rem_spe_texts)
        # 大文字小文字統一
        nor_texts = self.normalize_text(rem_spe_cha_texts)
        # 検索用文字列をベクトル変換
        embedding = self.embed_sentence(nor_texts)

        # 総当りで類似度を計算
        results = map(
                lambda i: {
                    'title': i['title'],
                    'body': i['body'],
                    # ここでクエリと各文章のコサイン類似度を計算
                    'similarity': self.cosine_similarity(i['embedding'], embedding)
                    },
                vector_database
        )
        # コサイン類似度で降順（大きい順）にソート
        results = sorted(results, key=lambda i: i['similarity'], reverse=True)

        return results[0]

if __name__ == "__main__":
    # 現在のディレクトリ取得
    dirname = os.getcwd()
    # モデルファイル
    saved_model = dirname + '/models/proto_model'
    cos_sim_calc = CosSimCalc(model_name=saved_model)
    cos_sim_calc.main(question = '商標権について')
