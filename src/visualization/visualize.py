import pandas as pd

from nltk.probability import FreqDist

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

comcast_complaints = pd.read_csv(
    "../../data/processed/proc_comcast-consumer-complaints.csv"
)

# ―――――――――――――――――――――――――――――――――――――
# DISTRIBUTION OF ALL CONSUMER RATINGS
# ―――――――――――――――――――――――――――――――――――――
sns.histplot(comcast_complaints["rating"])
plt.title("Distribution of Consumer Ratings")
plt.savefig("../visualization/figures/distribution-all-ratings.png")

# ――――――――――――――――――――――――――――――――――――
# DISTRIBUTION OF COMPLAINTS BY STATE
# ――――――――――――――――――――――――――――――――――――
plt.figure(figsize=(12, 10))
sns.histplot(y=comcast_complaints["author_state"])
plt.title("Distribution of Complaints by State")
plt.savefig("../visualization/figures/distribution-author-states.png")

# ―――――――――――――――――――――――――――――――――――――――――
# DISTRIBUTION OF CONSUMER RATINGS BY YEAR
# ―――――――――――――――――――――――――――――――――――――――――
ratings_by_year = pd.DataFrame(
    comcast_complaints.groupby("posted_year")["rating"].value_counts()
).reset_index()
g = sns.FacetGrid(ratings_by_year, col="posted_year", col_wrap=4, sharex=False)
g.map(sns.barplot, "rating", "count", log=True)
g.fig.suptitle("Distribution of Ratings by Year", y=1.03)
g.fig.tight_layout()
plt.savefig("../visualization/figures/distribution-ratings-by-year.png")

# ―――――――――――――――――――――――――――――――――――――――――
# DISTRIBUTION OF CONSUMER RATINGS BY DAY
# ―――――――――――――――――――――――――――――――――――――――――
ratings_by_day = pd.DataFrame(
    comcast_complaints.groupby("posted_day_name")["rating"].value_counts()
).reset_index()
g = sns.FacetGrid(ratings_by_day, col="posted_day_name", col_wrap=3, sharex=False)
g.map(sns.barplot, "rating", "count", log=True)
g.fig.suptitle("Distribution of Ratings by Day", y=1.03)
g.fig.tight_layout()
plt.savefig("../visualization/figures/distribution-ratings-by-day.png")

# ―――――――――――――――――――――――――――――――――――――――――
# DISTRIBUTION OF CONSUMER RATINGS BY DAY
# ―――――――――――――――――――――――――――――――――――――――――
ratings_by_state = pd.DataFrame(
    comcast_complaints.groupby("author_state")["rating"].value_counts()
).reset_index()
g = sns.FacetGrid(ratings_by_state, col="author_state", col_wrap=3, sharex=False)
g.map(sns.barplot, "rating", "count", log=True)
g.fig.suptitle("Distribution of Ratings by State", y=1.03)
g.fig.tight_layout()
plt.savefig("../visualization/figures/distribution-ratings-by-state.png")


# ――――――――――――――――――――――――――――――――――――――――――――――――
# WORDCLOUD OF CONSUMER COMPLAINTS - CLEANED TEXT
# ――――――――――――――――――――――――――――――――――――――――――――――――
wc = WordCloud(width=800, height=400, background_color="white").generate(
    " ".join(comcast_complaints["cleaned_text"])
)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of Consumer Complaints")
plt.savefig("../visualization/figures/wc-consumer-complaints.png")

# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# LOOKING AT WORD FREQUENCIES OF NEGATIVE REVIEWS (RATING < 3)
# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
word_freq = FreqDist(
    comcast_complaints.query("rating < 3")["cleaned_text"].str.split(" ").sum()
)
word_freq_df = pd.DataFrame(word_freq.items(), columns=["word", "count"])
word_freq_df.head(20).sort_values(by="count").plot(kind="barh", x="word", y="count")
plt.title("Top 20 Word Counts of Negative Reviews (rating < 3)")
plt.savefig("../visualization/figures/wfreq-negative-ratings.png")
