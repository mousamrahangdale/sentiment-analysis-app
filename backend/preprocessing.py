"""
Text preprocessing shared by inference paths.

IMPORTANT: this is copied verbatim from the training notebook
(ChatGPT_Sentiment_Analysis_file.ipynb, `clean_text`). The DistilBERT model
was trained on text cleaned this way, so inference MUST use the identical
function or accuracy will silently degrade.
"""

import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#\w+", "", text)         # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)     # keep letters + spaces only
    return text.strip()
