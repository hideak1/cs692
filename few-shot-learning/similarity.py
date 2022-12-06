import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import copy

stopwords = stopwords.words('english')


def clean_text(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def find_topn(candidates_map, question, n):
    c_qs = copy.deepcopy(candidates_map['questions'])
    c_qs.insert(0, question)

    cleaned = list(map(clean_text, c_qs))

    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)

    csim_vec = csim[0][1:]
    res = sorted(range(len(csim_vec)), key = lambda sub: csim_vec[sub])[-n:]
    res_ques = [candidates_map['questions'][i] for i in res]
    res_ans = [candidates_map['answers'][i] for i in res]

    return {'questions': res_ques, 'answers': res_ans}

def find_topn_with_sameid(candidates_map, question, n):
    c_qs = copy.deepcopy(candidates_map['questions'])
    c_qs.insert(0, question)

    db_ids = candidates_map['db_id']

    cleaned = list(map(clean_text, c_qs))

    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)

    csim_vec = csim[0][1:]
    sorted_index = sorted(range(len(csim_vec)), key = lambda sub: csim_vec[sub], reverse = True)

    res_index = []
    currect_db_id = -1
    for i in sorted_index:
        if currect_db_id == -1:
            currect_db_id = db_ids[i]
            res_index.append(i)
        elif currect_db_id == db_ids[i]:
                res_index.append(i)
        if len(res_index) >= n:
            break

    csim_list = [csim_vec[i] for i in res_index]
    print(csim_list)
    res_ques = [candidates_map['questions'][i] for i in res_index]
    res_ans = [candidates_map['answers'][i] for i in res_index]

    return {'db_id': currect_db_id, 'questions': res_ques, 'answers': res_ans}