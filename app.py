from flask import Flask, request, jsonify
from flask_cors import CORS
import cos_sim_calc
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)
# 現在のディレクトリ取得
dirname = os.getcwd()
# モデルファイル
saved_model = dirname + '/models/proto_model'


@app.route('/process', methods=['POST'])
def process_question():
    data = request.get_json()
    query = data.get('query')
    logging.debug("Received query: %s", query)
    try:
        vector = cos_sim_calc.CosSimCalc(saved_model)
        result = vector.main(query)
        # 結果を JSON 形式で返す
        return jsonify(result)
    except Exception as e:
        logging.error("Error processing query: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
