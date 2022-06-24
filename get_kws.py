import json
import pandas as pd
import pke


data = pd.read_csv('italian_crime_news.csv', encoding='iso-8859-1')
#texts_by_cat = {
#        cat : [] for cat in data['newspaper_tag'].unique()
#        }
kws_by_cat = {
        cat : [] for cat in data['newspaper_tag'].unique()
        }

for i, r in data.iterrows():
    if i == 200:
        break
    cat = r['newspaper_tag']

    title = r['title'] if type(r['title']) == str else ''
    text = r['text'] if type(r['text']) == str else ''
    full_text = title + '\n' + text
    texts_by_cat[cat].append(full_text)

for cat in texts_by_cat:
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=' '.join(texts_by_cat[cat])[:100000], language='it')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    kw = extractor.get_n_best(n=20)

    kws_by_cat[cat] = kw

with open('example_results.json', 'w+') as f:
    json.dump(kws_by_cat, f)

