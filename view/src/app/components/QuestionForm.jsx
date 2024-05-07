'use client';

import { useState } from 'react';

export default function QuestionForm() {
  // サーバーからの結果を保持する状態変数
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = e.target.elements.question.value;

    // サーバーにPOSTリクエストを送る
    const response = await fetch('http://localhost:5000/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query: question })
    });

    // サーバーからのレスポンスをJSON形式で取得
    const data = await response.json();
    // 結果を状態変数に保存
    setResult({ title: data.title, body: data.body, similarity: data.similarity } || 'No result returned'); // デフォルトで"結果がない"と表示
    e.target.reset();
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          name="question"
          placeholder="質問を入力してください"
          rows={5}
        />
        <button type="submit">送信</button>
      </form>
      
      {/* サーバーからの結果を表示 */}
      <div>
        <h3>回答:</h3>
        <p><strong>タイトル:</strong> {result.title}</p>
        <p><strong>類似度:</strong> {result.similarity}</p>
        <p><strong>内容:</strong> {result.body}</p>
      </div>
    </div>
  )
}
