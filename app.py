from flask import Flask, request, jsonify
from flask_cors import CORS
import prompt_to_gpt
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

# 現在のディレクトリ取得
dirname = os.getcwd()
# ベクトルモデル名
embedding_model_name = 'paraphrase-multilingual-mpnet-base-v2'
# ベクトルモデルフォルダ
emb_local_dir = os.path.join(dirname, 'models', 'embedding_model')

# 環境変数からAPIキーを取得
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")

# ベクトルモデルをグローバル変数として保持
emb_model = None

def load_models():
    global emb_model
    logging.info("ベクトルモデルをロード中...")
    emb_model = prompt_to_gpt.load_or_download_embmodel(embedding_model_name, emb_local_dir)
    logging.info("ベクトルモデルのロードが完了しました。")

@app.route('/process', methods=['POST'])
def process_question():
    global emb_model
    data = request.get_json()
    query = data.get('query')
    file_count = data.get('file_count')
    logging.debug("Received query: %s", query)
    logging.debug("Received file_count: %s", file_count)
    try:
        if emb_model is None:
            raise ValueError("ベクトルモデルがロードされていません。")
        cos_sim_calc = prompt_to_gpt.CosSimCalc(api_key, emb_model)
        instruction = '要約してください'
        results = cos_sim_calc.main(query, file_count, instruction)
        return jsonify({'results': results})
    except Exception as e:
        logging.error("Error processing prompt: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    load_models()  # アプリケーション起動時にモデルをロード
    app.run(debug=True)
