import os
import re
import unicodedata
from typing import List, Tuple
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
import torch
import logging
logging.basicConfig(level=logging.DEBUG)


def load_or_download_genmodel(model_name: str, local_dir: str) -> tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, PreTrainedModel|SentenceTransformer]:
    # ローカルディレクトリが存在しない場合は作成
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # モデルがローカルに存在するかをチェック
    if not (os.path.exists(os.path.join(local_dir, 'config.json')) and
            os.path.exists(os.path.join(local_dir, 'tokenizer_config.json'))):
        # モデルが存在しない場合はダウンロードして保存
        print(f"Model not found locally. Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)
        print(f"Saving model and tokenizer to {local_dir}...")
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
    else:
        # モデルが存在する場合はローカルからロード
        print(f"Loading generative model and tokenizer from {local_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)
    
    return tokenizer, model

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
    # def __init__(self, gen_model: PreTrainedModel|SentenceTransformer, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, emb_model:SentenceTransformer):
    def __init__(self, emb_model:SentenceTransformer):
        # self.generative_model = gen_model
        # self.tokenizer = tokenizer
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

    def split_text(self, text, chunk_size=50):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def process_text_in_chunks(self, question, text, model, tokenizer):
        chunks = self.split_text(text, chunk_size=50)
        combined_result = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            model = model.to("cuda")

        for chunk in chunks:
            prompt = f"{question}\n\n{chunk}"
            print(prompt)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            print("CPU",input_ids[0])
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
                print("GPU",input_ids[0])
            
            # モデルを使って推論を行う
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):  # 混合精度モードに設定
                output = self.generative_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100, num_beams=3)
            print("推論できた！",output)
            result_text = tokenizer.decode(output[0], skip_special_tokens=True)
            combined_result += result_text + "\n"
        
        return combined_result

    def main(self, question):
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
        results = [
            {
                'title': item['title'],
                'body': item['body'],
                'similarity': self.cosine_similarity(item['embedding'], embedding)
            }
            for item in vector_database
        ]

        # コサイン類似度で降順（大きい順）にソート
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        text = sorted_results[0]['body']


        # トークナイザーとモデルを読み込む
        # tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        # elyza_model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)

        # プロンプトを設定
        # prompt = nor_texts
        # inputs = tokenizer(prompt, return_tensors='pt')
        # input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]

        # モデルを使って推論を行う
        # output = self.generative_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=3500)
        result = self.process_text_in_chunks(question, text, self.generative_model, self.tokenizer)

        # 結果を表示
        print(result)
        return result

if __name__ == '__main__':
    # 現在のディレクトリ取得
    dirname = os.getcwd()
    # 生成モデル名
    # gen_model_name = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct'
    gen_model_name = 'elyza/ELYZA-japanese-Llama-2-7b-instruct'    
    # 生成モデルフォルダ
    gen_local_dir = dirname + '/models/gen_model'
    # ベクトルモデル名
    emmbeding_model_name = 'paraphrase-multilingual-mpnet-base-v2'
    # ベクトルモデルフォルダ
    emb_local_dir = dirname + '/models/embedding_model'
    # 生成モデルをロードまたはダウンロード
    # tokenizer, gen_model = load_or_download_genmodel(gen_model_name, gen_local_dir)
    # モデル量子化
    # quantized_model = torch.quantization.quantize_dynamic(gen_model, {torch.nn.Linear}, dtype=torch.qint8)
    # print(tokenizer, quantized_model    )
    print("----------------------------")
    # ベクトルモデルをロードまたはダウンロード
    emb_model = load_or_download_embmodel(emmbeding_model_name, emb_local_dir)
    print(emb_model)
    # cos_sim_calc = CosSimCalc(gen_model, tokenizer, emb_model)
    cos_sim_calc = CosSimCalc(emb_model)
    cos_sim_calc.main(question = '著作権について教えて下さい。')
