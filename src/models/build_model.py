# ――――――――――――――――――――――――――――――――――――――――――――
# TOPIC MODELING OF CONSUMER COMPLAINTS - LDA
# ――――――――――――――――――――――――――――――――――――――――――――
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from bertopic import BERTopic

import matplotlib.pyplot as plt
import seaborn as sns


comcast_complaints = pd.read_csv(
    "../../data/processed/proc_comcast-consumer-complaints.csv"
)

# Assuming negative comments having a rating less than 3 and more positive ratings greater than 3.
comcast_texts_neg = comcast_complaints.query("rating < 3")

# TODO:
comcast_texts_pos = comcast_complaints.query("rating > 3")


# Most frequent terms in comments considered to have a negative rating.
tfidf_vec = TfidfVectorizer(stop_words="english", max_df=0.1)
X_neg = tfidf_vec.fit_transform(comcast_texts_neg["cleaned_text"].values)

# Looking at the top 10 most common words in negative comments.
# It looks like they are mostly related to comcast packages and possibly wait times when calling for support.
common_neg_words = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_))
top_10_common_neg_words = pd.DataFrame(
    [
        (w, common_neg_words[w])
        for w in sorted(common_neg_words, key=common_neg_words.get)
    ][:10],
    columns=["Word", "Idf"],
)
top_10_common_neg_words

plt.figure(figsize=(10, 8))
plt.style.use("ggplot")
sns.scatterplot(top_10_common_neg_words, x="Word", y="Idf")
plt.xticks(rotation=45)
plt.title("Top 10 Most Common Words in Negative Comments")
plt.savefig("../visualization/figures/top-10-common-words-neg-comments.png")

# LDA
count_vect = CountVectorizer(stop_words="english", max_df=0.1)

lda = LatentDirichletAllocation(
    n_components=10, random_state=822023, learning_method="online"
)

X_neg_count = count_vect.fit_transform(comcast_texts_neg["cleaned_text"].values)
X_neg_topics = lda.fit_transform(X_neg_count)
feature_names_neg = count_vect.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}")
    print(" ".join([feature_names_neg[i] for i in topic.argsort()[: -5 - 1 : -1]]))


customer_service_complaints = X_neg_topics[:, 8].argsort()[::-1]

# Looking at 10 customer complaints about calling in to speak to customer service.
for iter_idx, complaint_idx in enumerate(customer_service_complaints[:10]):
    print(comcast_texts_neg["text"][complaint_idx])

# ―――――――――――――――――――――――――――――
# TOPIC MODELING WITH BERTOPIC
# ―――――――――――――――――――――――――――――
bert_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
topics, probs = bert_model.fit_transform(comcast_texts_neg["cleaned_text"].values)

bert_model.get_topic_info()

# Comparing to the LDA model, look at topic 1 which looks similar to customer service complaints.
bert_model.get_topic(1)
customer_service_complaints = bert_model.get_document_info(
    comcast_texts_neg["text"]
).query("Topic == 1")

customer_service_complaints

bert_model.visualize_topics()
bert_model.visualize_barchart([1, 22, 32, 37])
