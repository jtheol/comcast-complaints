import re
import string

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

plt.style.use("ggplot")

comcast_complaints = pd.read_csv("../../data/raw/raw_comcast-consumer-complaints.csv")

# ――――――――――――――――――――――――――――――――――――――
# CREATING FEATURES BASED ON THE AUTHOR (Location, Name)
# ――――――――――――――――――――――――――――――――――――――
comcast_complaints["author_name"] = comcast_complaints["author"].apply(
    lambda x: x.split("of")[0]
)
comcast_complaints["author_location"] = comcast_complaints["author"].apply(
    lambda x: x.split("of")[1].lower()
)

state_mapping = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}

comcast_complaints["author_state"] = comcast_complaints["author_location"].apply(
    lambda x: [val.strip() for val in x.split(",")][1]
    if len(x.split(",")) > 1
    else None
)
comcast_complaints["author_state"] = comcast_complaints["author_state"].apply(
    lambda x: state_mapping[x] if x in state_mapping else x
)
comcast_complaints["author_state"] = comcast_complaints["author_state"].str.replace(
    r"\d+", ""
)
comcast_complaints["author_state"] = comcast_complaints["author_state"].str.replace(
    ".", ""
)
comcast_complaints["author_state"] = comcast_complaints["author_state"].apply(
    lambda x: x if len(str(x)) == 2 else None
)
comcast_complaints["author_city"] = comcast_complaints["author_location"].apply(
    lambda x: x.split(",")[0]
)

# ―――――――――――――――――――――――――――――――――
# CREATING THE TIME-BASED FEATURES
# ―――――――――――――――――――――――――――――――――
comcast_complaints["posted_on"] = pd.to_datetime(
    comcast_complaints["posted_on"], format="mixed"
)

comcast_complaints["posted_year"] = comcast_complaints["posted_on"].dt.year
comcast_complaints["posted_month"] = comcast_complaints["posted_on"].dt.month
comcast_complaints["posted_day"] = comcast_complaints["posted_on"].dt.day
comcast_complaints["posted_day_name"] = comcast_complaints["posted_on"].dt.day_name()

# Looking at complaints from 2000 to 2016
comcast_complaints["posted_year"].min()
comcast_complaints["posted_year"].max()

# Dropping some values due to the state names not being valid and were replaced with null values.
comcast_complaints.dropna(how="any", axis=0, inplace=True)
comcast_complaints.reset_index(drop=True, inplace=True)


# ――――――――――――――
# CLEANING TEXT
# ――――――――――――――
comcast_complaints["cleaned_text"] = comcast_complaints["text"].str.lower()

# Removing unicode characters
comcast_complaints["cleaned_text"] = comcast_complaints["cleaned_text"].apply(
    lambda x: re.sub(
        r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x
    )
)

# Removing stopwords
stop_words = set(stopwords.words("english"))
comcast_complaints["cleaned_text"] = comcast_complaints["cleaned_text"].apply(
    lambda x: " ".join(w for w in x.split(" ") if w not in stop_words)
)

# Removing punctuation
comcast_complaints["cleaned_text"] = comcast_complaints["cleaned_text"].str.translate(
    str.maketrans("", "", string.punctuation)
)


comcast_complaints.to_csv(
    "../../data/processed/proc_comcast-consumer-complaints.csv", index=False
)
