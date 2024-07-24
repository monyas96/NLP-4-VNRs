#pip install Pytorch sentencepiece Tensorflow torch tk pyqt5

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import re
import networkx as nx
import urllib.parse  # Import the urllib module
from PyPDF2 import PdfReader  # Correctly import PdfReader
import seaborn as sns
from PyPDF2 import PdfFileReader
import requests
from io import BytesIO
from transformers import pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from nltk.corpus import wordnet
import sacremoses # is a library that helps with text tokenization and detokenization, and it's recommended for use with some Hugging Face models.
nltk.download('wordnet')
# Step 1: Extract VNRs text
african_countries = ["Congo", "Equatorial Guinea", "Guinea", "Libya", "Mauritania", "Mauritius", "Namibia", "Sierra Leone", "Solomon Islands", "Eritrea", "Kenya", "Nigeria", "South Africa", "Uganda", "Zimbabwe"]

base_url = "https://hlpf.un.org/sites/default/files/vnrs/2024/VNR%202024%20{}%20Report.pdf"

def format_country_name(country):
    return urllib.parse.quote(country, safe='')

def download_pdf(country):
    formatted_country = format_country_name(country)
    url = base_url.format(formatted_country)
    print(f"Attempting to download from URL: {url}")
    response = requests.get(url)
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        print(f"PDF for {country} downloaded successfully.")
        return BytesIO(response.content)
    else:
        print(f"PDF for {country} not found at URL: {url}.")
        return None

def extract_text_from_pdf(pdf_bytes):
    pdf = PdfReader(pdf_bytes)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ''
    return text

def load_data_from_vnrs(countries):
    texts = []
    for country in countries:
        pdf_bytes = download_pdf(country)
        if pdf_bytes:
            print(f"Extracting text from {country} PDF...")
            text = extract_text_from_pdf(pdf_bytes)
            print(f"Extracted text length for {country}: {len(text)}")
            if text:
                texts.append({'country': country, 'text': text})
            else:
                print(f"No text extracted from {country} PDF.")
        else:
            print(f"Failed to download PDF for {country}.")
    return pd.DataFrame(texts)

def main():
    print("Download and Extract VNR Key Messages")
    countries_to_process = african_countries
    data = load_data_from_vnrs(countries_to_process)
    if not data.empty:
        print(data)
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()

# Load data from VNRs
data = load_data_from_vnrs(african_countries)

# Save the DataFrame in your Python environment
df = pd.DataFrame(data)

#Text varification
# Check for empty texts
empty_texts = df[df['text'].str.strip() == '']
print(f"Number of empty texts: {len(empty_texts)}")
# Text length statistics
text_lengths = df['text'].str.len()
print(f"Text length statistics:\n{text_lengths.describe()}")


# Print samples of the extracted texts
sample_texts = df['text'].sample(5).values
print("Sample extracted texts:")
for i, text in enumerate(sample_texts, 1):
    print(f"\nSample {i}:\n{text[:500]}...")  # Print the first 500 characters of each sample
# Check for common phrases/words to verify text quality
common_phrases = ['sustainable development', 'poverty', 'health', 'education']
for phrase in common_phrases:
    occurrences = df['text'].str.contains(phrase, case=False).sum()
    print(f"Occurrences of '{phrase}': {occurrences}")
# Step 2: Define expanded keywords for each SDG
sdg_keywords = {
    'SDG 1: No Poverty': ['poverty', 'poor', 'income', 'financial aid', 'welfare', 'inequality', 'basic needs', 'social protection', 'unemployment', 'economic security'],
    'SDG 2: Zero Hunger': ['hunger', 'food security', 'nutrition', 'agriculture', 'farming', 'crop yield', 'malnutrition', 'food production', 'sustainable agriculture', 'food distribution'],
    'SDG 3: Good Health and Well-being': ['health', 'well-being', 'medical', 'disease', 'healthcare', 'hospital', 'mental health', 'vaccination', 'public health', 'life expectancy'],
    'SDG 4: Quality Education': ['education', 'school', 'literacy', 'training', 'teacher', 'universal education', 'learning', 'curriculum', 'early childhood education', 'inclusive education'],
    'SDG 5: Gender Equality': ['gender', 'women', 'equality', 'empowerment', 'discrimination', 'gender violence', 'female participation', 'gender rights', 'women leadership', 'gender parity'],
    'SDG 6: Clean Water and Sanitation': ['water', 'sanitation', 'clean water', 'hygiene', 'waste management', 'water access', 'water quality', 'safe drinking water', 'water supply', 'sanitary facilities'],
    'SDG 7: Affordable and Clean Energy': ['energy', 'renewable', 'electricity', 'power', 'solar', 'wind', 'energy access', 'clean energy', 'energy efficiency', 'sustainable energy'],
    'SDG 8: Decent Work and Economic Growth': ['work', 'employment', 'economic growth', 'jobs', 'labor', 'unemployment rate', 'workforce', 'job creation', 'economic development', 'fair wages'],
    'SDG 9: Industry, Innovation, and Infrastructure': ['industry', 'innovation', 'infrastructure', 'technology', 'industrialization', 'transport', 'resilient infrastructure', 'research and development', 'sustainable industrialization', 'innovation capacity'],
    'SDG 10: Reduced Inequality': ['inequality', 'equal', 'discrimination', 'inclusion', 'social justice', 'income disparity', 'equity', 'social inclusion', 'marginalized groups', 'equality of opportunity'],
    'SDG 11: Sustainable Cities and Communities': ['cities', 'communities', 'urban', 'sustainable', 'housing', 'urbanization', 'public spaces', 'urban planning', 'community resilience', 'transport systems'],
    'SDG 12: Responsible Consumption and Production': ['consumption', 'production', 'sustainable', 'resources', 'recycling', 'waste', 'sustainable production', 'consumer behavior', 'eco-friendly', 'circular economy'],
    'SDG 13: Climate Action': ['climate', 'global warming', 'carbon', 'emissions', 'climate change', 'mitigation', 'adaptation', 'carbon footprint', 'renewable energy', 'climate policy'],
    'SDG 14: Life Below Water': ['ocean', 'marine', 'sea', 'water', 'marine life', 'fisheries', 'marine ecosystems', 'ocean conservation', 'marine pollution', 'sustainable fisheries'],
    'SDG 15: Life on Land': ['forest', 'biodiversity', 'ecosystem', 'land', 'wildlife', 'reforestation', 'land degradation', 'habitat conservation', 'flora and fauna', 'deforestation'],
    'SDG 16: Peace and Justice Strong Institutions': ['peace', 'justice', 'institutions', 'law', 'human rights', 'rule of law', 'governance', 'conflict resolution', 'public institutions', 'security'],
    'SDG 17: Partnerships to achieve the Goal': ['partnership', 'cooperation', 'collaboration', 'support', 'development aid', 'global partnership', 'multilateralism', 'joint efforts', 'international cooperation', 'resource mobilization']
}

paradox_keywords = {
    'Financing Paradox': ['finance', 'funding', 'investment', 'economic', 'financial resources', 'budget', 'monetary', 'economic aid', 'capital', 'financial support'],
    'Energy Paradox': ['energy', 'electricity', 'power', 'renewable', 'solar energy', 'wind energy', 'energy access', 'clean energy', 'energy consumption', 'fossil fuels'],
    'Food Systems Paradox': ['food', 'agriculture', 'nutrition', 'hunger', 'food security', 'crop', 'food production', 'agricultural sustainability', 'food distribution', 'food systems']
}

# Expand keywords using WordNet
def expand_keywords(keywords):
    expanded_keywords = set(keywords)
    for word in keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name().replace('_', ' '))
    return list(expanded_keywords)

# Expand SDG and paradox keywords
sdg_keywords_expanded = {sdg: expand_keywords(keywords) for sdg, keywords in sdg_keywords.items()}
paradox_keywords_expanded = {paradox: expand_keywords(keywords) for paradox, keywords in paradox_keywords.items()}

# Load summarization and NER pipelines from Hugging Face transformers
pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Load translation pipeline
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Function to read and comprehend text
def read_and_comprehend(text, max_chunk_length=512):
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids'][0]

    chunks = [input_ids[i:i + max_chunk_length] for i in range(0, len(input_ids), max_chunk_length)]
    summaries = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = pipe(chunk_text, max_length=100, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)
# Function to translate text
def translate_text(text, max_chunk_length=512):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids'][0]

    chunks = [input_ids[i:i + max_chunk_length] for i in range(0, len(input_ids), max_chunk_length)]
    translations = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        translation = translation_pipeline(chunk_text, max_length=400)  # Adjusted max_length to avoid error
        translations.append(translation[0]['translation_text'])

    return ' '.join(translations)
# Function to identify themes
def identify_themes(text):
    ner_results = ner(text)
    themes = []
    for entity in ner_results:
        if entity['entity'] in ['B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-TIME', 'I-TIME', 'B-DATE', 'I-DATE']:
            themes.append((entity['entity'], entity['word']))
    return themes
# Function to extract numeric sentences
def extract_numeric_sentences(text):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    numeric_sentences = [sentence for sentence in sentences if re.search(r'\d+', sentence)]
    return numeric_sentences
# Function to categorize sentences
def categorize_sentence(sentence,sdg_keywords_expanded, paradox_keywords_expanded):
    sdg_code = None
    paradox_code = None

    for sdg, keywords in sdg_keywords_expanded.items():
        if any(keyword in sentence.lower() for keyword in keywords):
            sdg_code = sdg
            break

    for paradox, keywords in paradox_keywords_expanded.items():
        if any(keyword in sentence.lower() for keyword in keywords):
            paradox_code = paradox
            break

    return sdg_code, paradox_code

# Function to classify sentences

def classify_sentences(sentences, sdg_keywords_expanded, paradox_keywords_expanded):
    classified = []
    for sentence in sentences:
        sdg_code, paradox_code = categorize_sentence(sentence, sdg_keywords_expanded, paradox_keywords_expanded)
        if 'increase' in sentence.lower() or 'improvement' in sentence.lower() or 'progress' in sentence.lower():
            classified.append((sentence, 'Improvement', 1, sdg_code, paradox_code))
        elif 'decrease' in sentence.lower() or 'decline' in sentence.lower() or 'challenge' in sentence.lower():
            classified.append((sentence, 'Challenge', 0, sdg_code, paradox_code))
        else:
            classified.append((sentence, 'Other', None, sdg_code, paradox_code))
    return classified

# df is the DataFrame containing the VNR text data
analysis_results = {
    'VNR Year': [],
    'Country': [],
    'Code (SDG# Title)': [],
    'Quotation Exemplifying the Code': [],
    'Explanation (Challenge/Improvement)': [],
    'Dummy Variable': [],
    'Translation (if needed)': [],
    'Paradox Code': []
}

for index, row in df.iterrows():
    country = row['country']
    text = row['text']

    # Translate text if not in English
    if country in ["Guinea", "Mauritania", "Libya"]:  # Example for French and Arabic speaking countries
        text = translate_text(text)

    # Read and comprehend the text
    comprehended_text = read_and_comprehend(text)

    # Identify key themes and extract numeric sentences
    themes = identify_themes(comprehended_text)
    numeric_sentences = extract_numeric_sentences(comprehended_text)

    # Classify sentences into challenges or improvements with SDG and paradox categorization
    classified_sentences = classify_sentences(numeric_sentences, sdg_keywords_expanded, paradox_keywords_expanded)

    for sentence, classification, dummy, sdg_code, paradox_code in classified_sentences:
        if sdg_code:
            analysis_results['VNR Year'].append(2024)  # Example year
            analysis_results['Country'].append(country)
            analysis_results['Code (SDG# Title)'].append(sdg_code)
            analysis_results['Quotation Exemplifying the Code'].append(sentence)
            analysis_results['Explanation (Challenge/Improvement)'].append(classification)
            analysis_results['Dummy Variable'].append(dummy)
            analysis_results['Translation (if needed)'].append("")  # Add translation if needed
            analysis_results['Paradox Code'].append(paradox_code)

# Create DataFrame from the collected data
analysis_df = pd.DataFrame(analysis_results)

# Display the DataFrame
print(analysis_df)