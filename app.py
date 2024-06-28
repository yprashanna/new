from flask import Flask, request, jsonify
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import json
import os

app = Flask(__name__)

def load_data():
    with open('compex_scholarship_data.json', 'r') as f:
        data = json.load(f)

    # Prepare the question-answer pairs
    qa_pairs = []
    for course in data['courses']:
        question = f"What are the eligibility criteria for {course['course']}?"
        answer = course['eligibility']
        qa_pairs.append({"question": question, "answer": answer})

    qa_pairs.extend([
        {"question": "What is the age limit for COMPEX Scholarship?", "answer": data['age_limit']},
        {"question": "When is the COMPEX Scholarship announced?", "answer": data['application_process']['announcement']},
        {"question": "Where can I find the notice for the COMPEX Scholarship?", "answer": data['application_process']['notice_publication']},
        {"question": "What are the benefits of the COMPEX Scholarship?", "answer": "; ".join(data['scholarship_details']['benefits'])},
        {"question": "Do I have to pay for food under the COMPEX Scholarship?", "answer": data['scholarship_details']['mess_fees']},
        {"question": "How long does the college allocation process take?", "answer": data['college_allocation']['process_time']},
        {"question": "Who allocates the colleges?", "answer": data['college_allocation']['allocating_body']}
    ])
    return qa_pairs

qa_pairs = load_data()

# Load the pre-trained BERT model and tokenizer
mmodel_name = 'distilbert-base-uncased'
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# Initialize the pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    context = data.get('context')
    question = data.get('question')
    if not context or not question:
        return jsonify({'error': 'Invalid input'}), 400
    
    result = qa_pipeline({'context': context, 'question': question})
    return jsonify({'answer': result['answer']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
