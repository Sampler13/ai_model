from flask import Flask, render_template, request
from io import StringIO
from answering2 import generate_answer, MLModels, VectorDatabase, load_documents_from_folder
from collections import deque

app = Flask(__name__)

# Инициализируем модели и базу данных
models = MLModels()
vector_db = VectorDatabase(models.vector_model)
contexts = load_documents_from_folder('docs')
vector_db.populate(contexts)

# История чата
history = deque(maxlen=10)

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question']
        history.append(question)
        render_template('chat.html', history=history)
        answer = generate_answer(
            prompt='Используя этот контекст, ответь пожалуйста на мой вопрос. Постарайся дать максимально развернутый ответ, подходящий по смыслу:',
            question=question,
            history=list(history),
            tokenizer=models.ru_llm_tokenizer,
            config=models.ru_llm_config,
            model=models.ru_llm_model,
            vector_db=vector_db
        )
        
        history.append(answer)
        return render_template('chat.html', history=history)
    return render_template('chat.html', history=history)

@app.route('/save', methods=['POST'])
def save_chat():
    chat_content = StringIO()
    for message in history:
        chat_content.write(message + '\n')
    chat_content.seek(0)
    return chat_content.read(), 200, {'Content-Disposition': 'attachment;filename=chat_history.doc', 'Content-Type': 'application/msword'}

if __name__ == '__main__':
    app.run(debug=True)
