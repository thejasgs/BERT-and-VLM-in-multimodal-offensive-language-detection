import re

class TextCleaner():
    '''
    A simple text cleaner class to preprocess text data for meme classification.
    '''
    def __init__(self):
        pass
    
    def clean_text(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace('.gif', '').replace('jpg', '').replace('jpeg', '')
        return text
    
# Example usage:
# cleaner = TextCleaner()
# df_train_val['text'] = df_train_val['text'].apply(cleaner.clean_text)
# df_test['text'] = df_test['text'].apply(cleaner.clean_text)