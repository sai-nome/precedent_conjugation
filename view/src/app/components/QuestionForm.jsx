'use client';

export default function QuestionForm() {
  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = e.target.elements.question.value;
    const response = await fetch('http://localhost:5000/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query: question })
    });
    const data = await response.json();
    console.log(data);  // ここでサーバーからのレスポンスをログに出力
    e.target.reset();
  }

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        name="question"
        placeholder="質問を入力してください"
        rows={5}
      />
      <button type="submit">送信</button>
    </form>
  )
}
