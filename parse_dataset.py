import os
import csv
import sys
import collections
from bs4 import BeautifulSoup
import mailparser
import nltk

nltk.data.path.append("F://NLTK//")
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer

csv.field_size_limit(10 ** 9)


def parse_eml_dataset(rootDir):
    eml_counters = []
    all_words = set()
    words_count = 0
    emails_count = 0
    file_paths = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        print("Directory: %s" % dirName)
        for fname in fileList:
            try:
                with open(dirName + "\\" + fname, "r") as eml_file:
                    raw_email = eml_file.read()
                mail = mailparser.parse_from_string(
                    raw_email,
                )
                text_content = (
                    mail.subject
                    + " "
                    + BeautifulSoup(mail.body, features="html.parser").get_text()
                )
                text_no_punct = "".join(
                    [i for i in text_content if i.isalpha() or i.isspace()]
                )
                words_freq = collections.Counter(text_no_punct.lower().split())
                eml_counters += [words_freq]
                all_words.update(text_no_punct.lower().split())
                words_count += len(text_no_punct.lower().split())
                emails_count += 1
                file_paths += [dirName + "\\" + fname]
            except:
                print(fname + " is problematic to read/parse (wrong file format/etc")
    print(f"words in dataset ({rootDir}): {words_count} across {emails_count} emails")
    return eml_counters, all_words, file_paths


def parse_ham_dataset(full_file_path):
    counters = []
    labels = []
    words = set()
    words_count = 0
    emails_count = 0
    file_paths = []
    with open(full_file_path, newline="", encoding="utf-8") as csvfile:
        data = csv.reader(
            csvfile,
        )
        next(data)
        for row in data:
            words_freq = collections.Counter(row[0].lower().split())
            words_count += len(row[0].lower().split())
            emails_count += 1
            counters += [words_freq]
            labels += ["spam" if row[1] == "1" else "ham"]
            words.update(row[0].lower().split())
            file_paths += ["Kaggle " + str(emails_count)]
    print(
        f"words in dataset({full_file_path}): {words_count} across {emails_count} emails"
    )
    return counters, words, labels, file_paths


def save_data_for_som_as_csv(
    counters,
    features,
    labels,
    name="phishing_mail_frequency_matrix",
    spam_limit=9000,
    ham_limit=9000,
    phishing_limit=9000,
    min_amount_of_words=100,
    paths=[],
):
    os.chdir("F:\\_MAGISTRAS\\filtered_datasets\\")
    porter = PorterStemmer()
    stop_words = [s.replace("'", "") for s in stopwords.words("english")]
    labels_to_add = []
    paths_to_add = []

    with open(f"{name}.csv", "w", encoding="utf-8", newline="") as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(features)
        for x in range(len(counters)):
            counter = counters[x]
            new_counter = collections.Counter()
            for k in counter:
                if k not in stop_words:
                    new_counter[porter.stem(k)] += counter[k]
            r = []
            for f in features:
                r += [new_counter[f] if f in new_counter.keys() else 0]
            if sum(r) >= min_amount_of_words:
                if labels[x] == "ham":
                    ham_limit -= 1
                    if ham_limit < 0:
                        continue
                if labels[x] == "spam":
                    spam_limit -= 1
                    if spam_limit < 0:
                        continue
                if labels[x] == "phishing":
                    phishing_limit -= 1
                    if phishing_limit < 0:
                        continue
                wr.writerow(r)
                labels_to_add += [labels[x]]
                paths_to_add += [paths[x]]

    with open(f"{name}_labels.csv", "w", encoding="utf-8", newline="") as csvfile:
        wr = csv.writer(csvfile)
        for label in labels_to_add:
            wr.writerow(label)

    with open(f"{name}_paths.csv", "w", encoding="utf-8", newline="") as csvfile:
        wr = csv.writer(csvfile)
        for label in paths_to_add:
            wr.writerow(label)


stop_words = [s.replace("'", "") for s in stopwords.words("english")] + [
    "number",
    "url",
]

eml_data_counters, words1, paths1 = parse_eml_dataset(
    "F:\\Google Drive\\VGTU Stuff\\_MAGISTRAS\\email datasets\\public_phishing"
)
spam_ham_counters, words2, labels, paths2 = parse_ham_dataset(
    "F:\\Google Drive\\VGTU Stuff\\_MAGISTRAS\\email datasets\\spam+ham\\spam_or_not_spam.csv"
)
eml_data_counters2, words3, paths3 = parse_eml_dataset(
    "F:\\Google Drive\\VGTU Stuff\\_MAGISTRAS\\email datasets\\SPAM archive"
)

words = words1.union(words2).union(words3)
porter = PorterStemmer()

print(len(words))
all_words_freq = collections.Counter()
for counter in spam_ham_counters:
    all_words_freq += counter
for counter in eml_data_counters:
    all_words_freq += counter
for counter in eml_data_counters2:
    all_words_freq += counter

for min_frequency in [1, 5, 25, 100, 500, 1000]:
    features = set()
    for w in words:
        if (
            w not in stop_words
            and all_words_freq[w] > min_frequency
            and wordnet.synsets(w)
        ):
            features.add(porter.stem(w))

    for email_limit in [100000]:
        for min_amount_of_words in [1, 5, 10, 25]:

            save_data_for_som_as_csv(
                spam_ham_counters + eml_data_counters + eml_data_counters2,
                features,
                labels
                + ["phishing"] * len(eml_data_counters)
                + ["spam"] * len(eml_data_counters2),
                name=f"{email_limit}_sz_{min_frequency}_fr_{min_amount_of_words}_w",
                spam_limit=email_limit,
                ham_limit=email_limit,
                phishing_limit=email_limit,
                min_amount_of_words=min_amount_of_words,
                paths=paths2 + paths1 + paths3,
            )
