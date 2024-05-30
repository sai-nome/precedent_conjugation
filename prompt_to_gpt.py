import os
import re
import unicodedata
from typing import List, Tuple
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import torch
import logging
import os
from openai import OpenAI
logging.basicConfig(level=logging.INFO)

def load_or_download_embmodel(model_name: str, local_dir: str) -> SentenceTransformer:
    # ローカルディレクトリが存在しない場合は作成
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # モデルがローカルに存在するかをチェック
    if not (os.path.exists(os.path.join(local_dir, 'config.json')) and
            os.path.exists(os.path.join(local_dir, 'modules.json')) and
            os.path.exists(os.path.join(local_dir, 'tokenizer_config.json'))):
        # モデルが存在しない場合はダウンロードして保存
        print(f"Model not found locally. Downloading {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"Saving model and tokenizer to {local_dir}...")
        model.save(local_dir)
    else:
        # モデルが存在する場合はローカルからロード
        print(f"Loading embedding model and tokenizer from {local_dir}...")
        model = SentenceTransformer(local_dir)
    
    return model

class CosSimCalc:
    def __init__(self, api_key:str, emb_model:SentenceTransformer):
        self.open_api_key = api_key
        self.embedding_model = emb_model

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
            # モデルでベクトル化
            vector = self.embedding_model.encode(segment, convert_to_tensor=True)
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

    def cosine_similarity(self, v1, v2) -> int:
        # ベクトルのドット積
        dot_product = np.dot(v1, v2)
        # ベクトルのノルム
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # コサイン類似度
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity

    def generate_response(self, prompt: str, max_tokens: int = 3000) -> str:
        try:
            client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            response = client.chat.completions.create(
                model="gpt-4o",  # 使用するモデルを指定
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0.7
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"エラーが発生しました: {e}")
            return "申し訳ありませんが、応答を生成する際にエラーが発生しました。"

    def process_long_text(self, instruction: str, prompt: str, chunk_size: int = 1000) -> str:
        words = prompt.split()
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        responses = []

        for chunk in chunks:
            chunk_prompt = f"{instruction}\n\n{chunk}"
            response = self.generate_response(chunk_prompt)
            responses.append(response)

        return ' '.join(responses)

    def main(self, question, file_count, instruction):
        # 現在のディレクトリ取得
        dirname = os.getcwd()
        # ベクトルフォルダ
        vector_fol = dirname + '/vector_database'
        # ベクトルデータベース
        vector_db_json = vector_fol + '/vector_database.json'
        with open(vector_db_json, 'r', encoding='utf-8') as file:
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
        results = []
        for item in vector_database:
            try:
                # print(item[0])
                similarity = self.cosine_similarity(item['embedding'], np.array(embedding))
                title = item['title']
                body = item['body']
                results.append({'title': title, 'body': body, 'similarity': similarity})
                results = sorted(results, key=lambda x: x['similarity'], reverse=True)
            except:
                continue
        

        returned_items = []
        for i in range(file_count):
            print(file_count)
            text = results[i]['body']

            # モデルを使って推論を行う
            inference = self.process_long_text(instruction, text)

            # 結果を表示
            returned_item = {'title': results[i]['title'],
                    'body': results[i]['body'],
                    'similarity': results[i]['similarity'],
                    'summary': inference
                    }
            returned_items.append(returned_item)
        return returned_items

if __name__ == '__main__':
    # 環境変数からAPIキーを取得
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")
    # 現在のディレクトリ取得
    dirname = os.getcwd()
    # ベクトルモデル名
    emmbeding_model_name = 'paraphrase-multilingual-mpnet-base-v2'
    # ベクトルモデルフォルダ
    emb_local_dir = dirname + '/models/embedding_model'
    # ベクトルモデルをロードまたはダウンロード
    emb_model = load_or_download_embmodel(emmbeding_model_name, emb_local_dir)
    cos_sim_calc = CosSimCalc(api_key, emb_model)
    question = '著作権について教えて下さい。'
    file_count = 3
    instruction = '要約してください'
    cos_sim_calc.main(question, file_count, instruction)
