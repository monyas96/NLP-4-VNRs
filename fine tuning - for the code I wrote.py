import pandas as pd
import openpyxl
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import transformers
import accelerate
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the training data from the Excel files
df_sdg = pd.read_excel("C:/Users/MYASSIEN/OneDrive - United Nations/Shared Documents - SMU data team/01. OSAA projects/06.2024_NLP for VNRs/Training data _SDG.xlsx")
df_paradox = pd.read_excel("C:/Users/MYASSIEN/OneDrive - United Nations/Shared Documents - SMU data team/01. OSAA projects/06.2024_NLP for VNRs/Paradox analysis_training data.xlsx")

# Adjust these variable names if they differ from your actual column names
text_column = 'Quote'
sdg_label_column = 'SDGcode'
paradox_label_column = 'ParadoxCode'
explanation_column = 'Explanation'

# Check the columns in the dataframes

print("SDG DataFrame columns:", df_sdg.columns)
print("Paradox DataFrame columns:", df_paradox.columns)

# Define label mappings
sdg_label_mapping = {
    'SDG#1 No Poverty': 1,
    'SDG#2 Zero Hunger': 2,
    'SDG#3 Good Health and Well-being': 3,
    'SDG#4 Quality Education': 4,
    'SDG#5 Gender Equality': 5,
    'SDG#6 Clean Water and Sanitation': 6,
    'SDG#7 Affordable and Clean Energy': 7,
    'SDG#8 Decent Work and Economic Growth': 8,
    'SDG#9 Industry, Innovation, and Infrastructure': 9,
    'SDG#10 Reduced Inequality': 10,
    'SDG#11 Sustainable Cities and Communities': 11,
    'SDG#12 Responsible Consumption and Production': 12,
    'SDG#13 Climate Action': 13,
    'SDG#14 Life Below Water': 14,
    'SDG#15 Life on Land': 15,
    'SDG#16 Peace, Justice, and Strong Institutions': 16,
    'SDG#17 Partnerships for the Goals': 17
}

paradox_label_mapping = {
    'Financing Paradox': 1,
    'Energy Paradox': 2,
    'Food Systems Paradox': 3
}

explanation_label_mapping = {
    'Improvement': 1,
    'Challenge': 0
}

# Ensure the columns exist in the DataFrame
if sdg_label_column in df_sdg.columns:
    df_sdg['SDG_Label'] = df_sdg[sdg_label_column].map(sdg_label_mapping)
else:
    print(f"Column '{sdg_label_column}' not found in SDG DataFrame.")

if paradox_label_column in df_paradox.columns:
    df_paradox['Paradox_Label'] = df_paradox[paradox_label_column].map(paradox_label_mapping)
else:
    print(f"Column '{paradox_label_column}' not found in Paradox DataFrame.")

# Ensure the explanation column exists in both dataframes
if explanation_column not in df_sdg.columns:
    print(f"Column '{explanation_column}' not found in SDG DataFrame.")
if explanation_column not in df_paradox.columns:
    print(f"Column '{explanation_column}' not found in Paradox DataFrame.")

# Print first few rows to verify data
print("First few rows of SDG DataFrame:")
print(df_sdg.head())

print("First few rows of Paradox DataFrame:")
print(df_paradox.head())

# Split the Explanation column into Explanation_Label and Detailed_Explanation
def split_explanation(df, explanation_column):
    df[['Explanation_Label', 'Detailed_Explanation']] = df[explanation_column].str.split(':', n=1, expand=True)
    df['Explanation_Label'] = df['Explanation_Label'].str.strip()
    df['Detailed_Explanation'] = df['Detailed_Explanation'].str.strip()
    df['Explanation_Label'] = df['Explanation_Label'].map(explanation_label_mapping)
    return df

df_sdg = split_explanation(df_sdg, explanation_column)
df_paradox = split_explanation(df_paradox, explanation_column)

# Combine the dataframes (only SDG_Label in SDG dataframe and Paradox_Label in Paradox dataframe)
df_sdg = df_sdg[[text_column, 'SDG_Label', 'Detailed_Explanation', 'Explanation_Label']].dropna()
df_paradox = df_paradox[[text_column, 'Paradox_Label', 'Detailed_Explanation', 'Explanation_Label']].dropna()

# Add empty columns to match the dataframes
if 'Paradox_Label' not in df_sdg.columns:
    df_sdg['Paradox_Label'] = None
if 'SDG_Label' not in df_paradox.columns:
    df_paradox['SDG_Label'] = None

# Combine into a single DataFrame for training
df = pd.concat([df_sdg, df_paradox], ignore_index=True)

# Check for NaN values in the combined DataFrame
print("Checking for NaN values in the combined DataFrame:")
print(df.isna().sum())

# Check the DataFrame after preprocessing
print("DataFrame after preprocessing:")
print(df.head())

# Preprocess data
def preprocess_data(df):
    df = df.dropna(subset=[text_column, 'Explanation_Label'])
    df[text_column] = df[text_column].astype(str)
    df['SDG_Label'] = df['SDG_Label'].fillna(-1).astype(int)  # Fill NA with -1 for SDG_Label
    df['Paradox_Label'] = df['Paradox_Label'].fillna(-1).astype(int)  # Fill NA with -1 for Paradox_Label
    df['Explanation_Label'] = df['Explanation_Label'].astype(int)
    return df

df = preprocess_data(df)

# Check the DataFrame after dropping NA
print("DataFrame after dropping NA:")
print(df.head())
print(f"Number of samples: {len(df)}")

# Ensure there's data before splitting
if len(df) == 0:
    raise ValueError("No data available after preprocessing. Please check your data.")

# Split the data
# Split the data
train_texts, val_texts, train_sdg_labels, val_sdg_labels, train_paradox_labels, val_paradox_labels, train_explanation_labels, val_explanation_labels = train_test_split(
    df[text_column].tolist(),
    df['SDG_Label'].tolist(),
    df['Paradox_Label'].tolist(),
    df['Explanation_Label'].tolist(),
    test_size=0.2
)
# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create torch datasets
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sdg_labels, paradox_labels, explanation_labels):
        self.encodings = encodings
        self.sdg_labels = sdg_labels
        self.paradox_labels = paradox_labels
        self.explanation_labels = explanation_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['sdg_labels'] = torch.tensor(self.sdg_labels[idx])
        item['paradox_labels'] = torch.tensor(self.paradox_labels[idx])
        item['explanation_labels'] = torch.tensor(self.explanation_labels[idx])
        return item

    def __len__(self):
        return len(self.sdg_labels)

train_dataset = MultiLabelDataset(train_encodings, train_sdg_labels, train_paradox_labels, train_explanation_labels)
val_dataset = MultiLabelDataset(val_encodings, val_sdg_labels, val_paradox_labels, val_explanation_labels)

# Define the model
# Define custom trainer to handle multiple labels
class MultiTaskBertModel(torch.nn.Module):
    def __init__(self, model_name, num_sdg_labels, num_paradox_labels, num_explanation_labels):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_sdg_labels)
        self.classifier_sdg = torch.nn.Linear(self.bert.config.hidden_size, num_sdg_labels)
        self.classifier_paradox = torch.nn.Linear(self.bert.config.hidden_size, num_paradox_labels)
        self.classifier_explanation = torch.nn.Linear(self.bert.config.hidden_size, num_explanation_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        sdg_logits = self.classifier_sdg(pooled_output)
        paradox_logits = self.classifier_paradox(pooled_output)
        explanation_logits = self.classifier_explanation(pooled_output)

        return sdg_logits, paradox_logits, explanation_logits
#Instantiate the model:
    model = MultiTaskBertModel(
        model_name='bert-base-uncased',
        num_sdg_labels=df['SDG_Label'].nunique(),
        num_paradox_labels=df['Paradox_Label'].nunique(),
        num_explanation_labels=df['Explanation_Label'].nunique()
    )

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"  # Updated parameter name
)

# Define custom trainer to handle multiple labels
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_sdg = inputs.pop("SDG_Labels")  # Corrected label name
        labels_paradox = inputs.pop("Paradox_Labels")  # Corrected label name
        labels_explanation = inputs.pop("Explanation_Labels")  # Corrected label name

        sdg_logits, paradox_logits, explanation_logits = model(**inputs)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss_sdg = loss_fct(sdg_logits.view(-1, model.classifier_sdg.out_features), labels_sdg.view(-1))
        loss_paradox = loss_fct(paradox_logits.view(-1, model.classifier_paradox.out_features), labels_paradox.view(-1))
        loss_explanation = loss_fct(explanation_logits.view(-1, model.classifier_explanation.out_features), labels_explanation.view(-1))

        loss = loss_sdg + loss_paradox + loss_explanation

        return (loss, (sdg_logits, paradox_logits, explanation_logits)) if return_outputs else loss

#Instantiate the Trainer:
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()