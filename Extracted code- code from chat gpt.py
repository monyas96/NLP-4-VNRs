
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import re
import fitz  # PyMuPDF

# Load NER pipeline from Hugging Face transformers
ner = pipeline("ner")

# Function to split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to identify key themes with focus on numeric findings
def identify_themes(text):
    ner_results = ner(text)
    themes = []
    for entity in ner_results:
        if entity['entity'] in ['B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-TIME', 'I-TIME', 'B-DATE', 'I-DATE']:
            themes.append((entity['entity'], entity['word']))
    return themes

# Function to extract sentences containing numeric findings
def extract_numeric_sentences(text):
    sentences = re.split(r'(?<!/w/./w.)(?<![A-Z][a-z]/.)(?<=/.|/?)/s', text)
    numeric_sentences = [sentence for sentence in sentences if re.search(r'/d+', sentence)]
    return numeric_sentences

# Function to classify sentences into challenges or improvements
def classify_sentences(sentences):
    classified = []
    for sentence in sentences:
        if 'increase' in sentence.lower() or 'improvement' in sentence.lower() or 'progress' in sentence.lower():
            classified.append((sentence, 'Improvement', 1))
        elif 'decrease' in sentence.lower() or 'decline' in sentence.lower() or 'challenge' in sentence.lower():
            classified.append((sentence, 'Challenge', 0))
        else:
            classified.append((sentence, 'Other', None))
    return classified

# Function to create word clouds
def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Function to create bar charts for SDG progress
def create_bar_chart(data, title):
    plt.figure(figsize=(10, 6))
    plt.bar(data['SDG'], data['Frequency'])
    plt.xlabel('SDG')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

# Function to create radial charts for SDG progress
def create_radial_chart(data, title):
    labels = data['SDG']
    values = data['Frequency']
    angles = [n / float(len(labels)) * 2 * 3.14 for n in range(len(labels))]
    angles += angles[:1]
    values += values[:1]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, values)
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(title)
    plt.show()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Example PDF path
pdf_path = "C:/Users/MYASSIEN/OneDrive - United Nations/Shared Documents - SMU data team/01. OSAA projects/06.2024_NLP for VNRs/VNRS/Zimbabwe - included - Viz/VNR 2024 Zimbabwe Main Messages.pdf"


# Extract text from PDF
extracted_text = extract_text_from_pdf(pdf_path)
print("Extracted Text:", extracted_text)

# Identify key themes and extract numeric sentences
themes = identify_themes(extracted_text)
print("Identified Themes:", themes)

numeric_sentences = extract_numeric_sentences(extracted_text)
print("Numeric Sentences:", numeric_sentences)

# Classify sentences into challenges or improvements
classified_sentences = classify_sentences(numeric_sentences)
print("Classified Sentences:", classified_sentences)

# Update DataFrame with the classified sentences
data = {
    'VNR Year': [],
    'Code (SDG# Title)': [],
    'Quotation Exemplifying the Code': [],
    'Explanation (Challenge/Improvement)': [],
    'Dummy Variable': [],
    'Translation (if needed)': []
}

for sentence, classification, dummy in classified_sentences:
    data['VNR Year'].append(2022)  # Example year
    data['Code (SDG# Title)'].append("SDG 8: Decent Work and Economic Growth")
    data['Quotation Exemplifying the Code'].append(sentence)
    data['Explanation (Challenge/Improvement)'].append(classification)
    data['Dummy Variable'].append(dummy)
    data['Translation (if needed)'].append("")  # Add translation if needed

# Create DataFrame from the collected data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Generate text for word clouds
text_improvement = " ".join([quote for quote, classification, _ in classified_sentences if classification == 'Improvement'])
text_challenges = " ".join([quote for quote, classification, _ in classified_sentences if classification == 'Challenge'])

# Create word clouds
create_word_cloud(text_improvement, "Areas of Improvement")
create_word_cloud(text_challenges, "Challenges")

# Sample data for bar chart (assuming aggregated frequency data is available)
sdg_data = {
    'SDG': ['SDG 1', 'SDG 8', 'SDG 4', 'SDG 16'],
    'Frequency': [5, 10, 7, 3]
}
sdg_df = pd.DataFrame(sdg_data)

# Create bar chart for SDG progress
create_bar_chart(sdg_df, "SDG Progress")

# Create radial chart for SDG progress
create_radial_chart(sdg_df, "SDG Progress")
