# Put your trained checkpoint here

This folder should contain the files produced by your notebook's:

```python
model.save_pretrained("saved_models/distilbert_sentiment")
tokenizer.save_pretrained("saved_models/distilbert_sentiment")
```

That is:

```
distilbert_sentiment/
├── config.json
├── model.safetensors        (or pytorch_model.bin)
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

Copy those exact files from wherever your notebook ran (Colab `saved_models/distilbert_sentiment/`,
downloaded as a zip, or synced from Drive) into this folder. The backend loads directly
from this path — no retraining needed.
