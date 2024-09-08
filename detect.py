import csv
import boto3
import json
import matplotlib.pyplot as plt
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def create_comprehend_client():
    return boto3.client(
        'comprehend',
        region_name='***',
        aws_access_key_id='***',
        aws_secret_access_key='***'
    )

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines=file.readlines()
        return lines[:]
def read_data_(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        combined_lines = [line1.strip() + "." + line2.strip() for line1, line2 in zip(lines1, lines2)]
        return combined_lines
def read_data__(file1_path, file2_path,file3_path):
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2,open(file3_path, 'r', encoding='utf-8') as file3:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        lines3 = file3.readlines()
        combined_lines = [line1.strip() + "" + line2.strip()+ line3.strip() for line1, line2,line3 in zip(lines1, lines2,lines3)]
        return combined_lines
def read_data_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            orig_sent = row['orig_sent0']
            aug_sent = row['aug_sent0']
            data.append(orig_sent+ aug_sent)
    return data


def process_and_save_data(texts, comprehend, filename, max_retries=3):
    scores = {label: [] for label in ['PROFANITY', 'HATE_SPEECH', 'INSULT', 'GRAPHIC', 'HARASSMENT_OR_ABUSE', 'SEXUAL', 'VIOLENCE_OR_THREAT', 'Toxicity']}
    with open(filename, 'a', encoding='utf-8') as file:
        for text in texts:
            attempts = 0
            while attempts < max_retries:
                try:
                    response = comprehend.detect_toxic_content(
                        TextSegments=[{'Text': text.strip()}],
                        LanguageCode='en'
                    )
                    json.dump(response, file)
                    file.write('\n')
                    for label in response['ResultList'][0]['Labels']:
                        scores[label['Name']].append(label['Score'])
                       # Handle the overall Toxicity score
                    toxicity_score = response['ResultList'][0].get('Toxicity', None)
                    if toxicity_score is not None:
                        scores['Toxicity'].append(toxicity_score)
                    break  
                except Exception as e:
                    attempts += 1
                    print(f"Error processing text, attempt {attempts}: {e}")
                    if attempts == max_retries:
                        print("Max retries reached, moving to next text.")
    return scores


def plot_histograms(scores):
    plt.figure(figsize=(10, 6))
    for i, (label, data) in enumerate(scores.items(), 1):
        plt.subplot(3, 3, i)
        plt.hist(data, bins=20, alpha=0.75, label=label)
        plt.title(label)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('female1.jpg')
    plt.show()

def read_scores_from_file(file_path):
    scores = {'PROFANITY': [], 'HATE_SPEECH': [], 'INSULT': [], 'GRAPHIC': [], 'HARASSMENT_OR_ABUSE': [], 'SEXUAL': [], 'VIOLENCE_OR_THREAT': [], 'Toxicity': []}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            response = json.loads(line)
            for label in response['ResultList'][0]['Labels']:
                scores[label['Name']].append(label['Score'])
            toxicity_score = response['ResultList'][0].get('Toxicity', None)
            if toxicity_score is not None:
                scores['Toxicity'].append(toxicity_score)
    return scores

def main():
    comprehend = create_comprehend_client()
    female_texts = read_data('gender/generate_female.txt')
    mix_texts=read_data_csv('entailment_data.csv')

    all_texts = mix_texts[:]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_and_save_data, all_texts[i:i+3], comprehend, 'auto_mix_responses.json') for i in range(0, len(all_texts), 3)]
        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
            results.append(f.result())

    final_scores = {label: [] for label in ['PROFANITY', 'HATE_SPEECH', 'INSULT', 'GRAPHIC', 'HARASSMENT_OR_ABUSE', 'SEXUAL', 'VIOLENCE_OR_THREAT', 'Toxicity']}
    for scores in results:
        for key, value in scores.items():
            final_scores[key].extend(value)

    plot_histograms(final_scores)

if __name__ == '__main__':
    main()

