from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
import language_tool_python
import spacy

app = Flask(__name__)

# Initialize the grammar checking tool and NLP model
tool = language_tool_python.LanguageTool('en-US')
nlp = spacy.load('en_core_web_sm')

# Directory where datasets are stored
data_dir = "Task7"

# Load datasets
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

preprocessed_train_set = load_data(os.path.join(data_dir, 'preprocessed_train_set.csv'))
preprocessed_valid_set = load_data(os.path.join(data_dir, 'preprocessed_valid_set.csv'))

# Preprocessing function
def preprocess_essay(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ''

# Function to check grammatical mistakes
def check_grammar(essay):
    matches = tool.check(essay)
    num_errors = len(matches)
    print(matches)
    return num_errors, matches

# Function to calculate the number of lines in the essay
def count_lines(essay):
    lines = essay.splitlines()
    num_lines = len([line for line in lines if line.strip()])  # Count non-empty lines
    return num_lines

# Function to extract the main concept of the essay using Named Entity Recognition (NER)
def extract_concept(essay):
    doc = nlp(essay)
    concepts = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'NORP', 'PRODUCT', 'EVENT']]
    return concepts if concepts else None

# Function to check if the essay relates to the main concept
def is_essay_relevant_to_concept(essay, concepts):
    if not concepts:
        return True  # If no concept was identified, we cannot penalize it

    doc = nlp(essay)
    matches = 0
    for concept in concepts:
        if concept.lower() in doc.text.lower():
            matches += 1

    total_sentences = len(list(doc.sents))
    if matches < total_sentences // 2:
        return False
    return True

# Function to fetch a score from the dataset based on a preprocessed essay
def get_score_from_dataset(essay):
    processed_essay = preprocess_essay(essay)
    
    if preprocessed_train_set is not None and processed_essay in preprocessed_train_set['essay'].values:
        matched_row = preprocessed_train_set[preprocessed_train_set['essay'] == processed_essay]
        return int(matched_row['score'].values[0])
    
    if preprocessed_valid_set is not None and processed_essay in preprocessed_valid_set['essay'].values:
        matched_row = preprocessed_valid_set[preprocessed_valid_set['essay'] == processed_essay]
        return int(matched_row['score'].values[0])

    return None

# Main scoring function with explanation
def score_essay(essay):
    explanation = []
    
    essay = preprocess_essay(essay)
    
    dataset_score = get_score_from_dataset(essay)
    if dataset_score is not None:
        explanation.append(f"Matched essay in the dataset with a score of {dataset_score}.")
        return dataset_score, explanation
    
    # Get the number of lines
    num_lines = count_lines(essay)
    explanation.append(f"The essay has {num_lines} lines.")

    # Get the number of grammatical errors
    num_grammar_errors, mac = check_grammar(essay)
    explanation.append(f"The essay contains {num_grammar_errors} grammatical errors.")

    # Base score initialization
    score = 3  # Start with a default score of 3

    # Adjust score based on the number of lines
    if num_lines >= 10:
        score += 2
        explanation.append("The essay has 10 or more lines, so 2 points were added.")
    elif num_lines >= 5:
        score += 1
        explanation.append("The essay has 5 to 9 lines, so 1 point was added.")
    else:
        score -= 1
        explanation.append("The essay has fewer than 5 lines, so 1 point was subtracted.")

    # Adjust score based on grammatical mistakes
    if num_grammar_errors > 10:
        score -= 2
        explanation.append("The essay has more than 10 grammatical errors, so 2 points were subtracted.")
    elif num_grammar_errors > 5:
        score -= 1
        explanation.append("The essay has 6 to 10 grammatical errors, so 1 point was subtracted.")
    else:
        score += 1
        explanation.append("The essay has 5 or fewer grammatical errors, so 1 point was added.")

    # Check if the essay is relevant to the identified concept
    concepts = extract_concept(essay)
    is_relevant = is_essay_relevant_to_concept(essay, concepts)
    
    if not is_relevant:
        score -= 3
        explanation.append("The essay is off-topic, so 3 points were subtracted.")

    # Ensure score is between 1 and 6
    score = max(1, min(score, 6))

    explanation.append(f"The final score is {score}.")
    return score, explanation

@app.route('/')
def index():
    return render_template('Task7.html')

@app.route('/submit', methods=['POST'])
def submit():
    essay = request.form.get('essay')
    if not essay:
        return jsonify({'error': 'No essay provided'}), 400

    try:
        score, explanation = score_essay(essay)
        return jsonify({'score': score, 'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
