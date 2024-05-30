'use client';

import { useState } from 'react';

export default function QuestionForm() {
  // サーバーからの結果を保持する状態変数
  const [results, setResults] = useState([]);
  const [fileCount, setFileCount] = useState(1); // 初期値を1に設定

  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = e.target.elements.question.value;

    // サーバーにPOSTリクエストを送る
    const response = await fetch('http://localhost:5000/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query: question, file_count: fileCount }) // file_countを追加
    });

    // サーバーからのレスポンスをJSON形式で取得
    const data = await response.json();
    
    // 結果にvisibleプロパティを追加して状態変数に保存
    const resultsWithVisibility = data.results.map(result => ({ ...result, visible: false }));
    setResults(resultsWithVisibility);
    e.target.reset();
  };

  const toggleVisibility = (index) => {
    setResults(results.map((result, i) => (
      i === index ? { ...result, visible: !result.visible } : result
    )));
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          name="question"
          placeholder="質問を入力してください"
          rows={5}
        />
        <br />
        <label htmlFor="fileCount">要約するファイルの個数:</label>
        <select
          name="fileCount"
          id="fileCount"
          value={fileCount}
          onChange={(e) => setFileCount(Number(e.target.value))} // 文字列を数値に変換
        >
          {[1, 2, 3, 4, 5].map(count => (
            <option key={count} value={count}>{count}</option>
          ))}
        </select>
        <br />
        <button type="submit">送信</button>
      </form>
      
      {/* サーバーからの結果を表示 */}
      <div>
        <h3>回答:</h3>
        {results.slice(0, 5).map((result, index) => (
          <div key={index}>
            <p><strong>タイトル:</strong> {result.title}</p>
            <p><strong>類似度:</strong> {result.similarity}</p>
            <p><strong>要約:</strong> {result.summary}</p>
            <p>
              <strong>内容:</strong> 
              <button onClick={() => toggleVisibility(index)}>
                {result.visible ? '▼' : '▶'}
              </button>
            </p>
            {result.visible && <p>{result.body}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
