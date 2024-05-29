from flask import Flask, request, jsonify
from flask_cors import CORS
# import cos_sim_calc
import prompt_to_gpt
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)
# 現在のディレクトリ取得
dirname = os.getcwd()
# モデルファイル
saved_model = dirname + '/models/proto_model'

# 環境変数からAPIキーを取得
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")
# ベクトルモデル名
emmbeding_model_name = 'paraphrase-multilingual-mpnet-base-v2'
# ベクトルモデルフォルダ
emb_local_dir = dirname + '/models/embedding_model'

@app.route('/process', methods=['POST'])
def process_question():
    data = request.get_json()
    query = data.get('query')
    logging.debug("Received query: %s", query)
    try:
        # ベクトルモデルをロードまたはダウンロード
        emb_model = prompt_to_gpt.load_or_download_embmodel(emmbeding_model_name, emb_local_dir)
        cos_sim_calc = prompt_to_gpt.CosSimCalc(api_key, emb_model)
        # query = '著作権について教えて下さい。'
        instruction = '要約してください'
        result = cos_sim_calc.main(query, instruction)
        # vector = cos_sim_calc.CosSimCalc(saved_model)
        # result = vector.main(query)
        # 結果を JSON 形式で返す
        return jsonify(result)
    except Exception as e:
        logging.error("Error processing query: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
