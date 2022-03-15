# import nltk
# nltk.download('wordnet')
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from nltk.corpus import stopwords


def word_tokenizer(review, without_punctuation=True):
    """
    Tokenizes the review into words after lowercasing. Returns a list of tokens
    Param:  review: review as string
            without_punctuation: Boolean, True if you want punctuation removed
    """
    review = review.lower()
    if without_punctuation == False:
        return wordpunct_tokenize(review)
    else:
        return RegexpTokenizer(r'\w+').tokenize(review)

# this one might be handy for POS-tagging
def sent_tokenizer(review):
    """
    Tokenizes the review into sentences
    """
    return sent_tokenize(review)

# for now only english words, might add other laguages later
def remove_stopwords(tokenized_review):
    """
    Removes english stop words from tokenized review
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokenized_review if not word in stop_words]

# I don't know whether stemming or lemmatizing would be better here
def stemmer(tokenized_review):
    """
    Stem the tokenized  review
    """
    return [PorterStemmer().stem(word) for word in tokenized_review]

def lemmatizer(tokenized_review):
    """
    Lemmatize the tokenized  review
    """
    return [WordNetLemmatizer().lemmatize(word) for word in tokenized_review]
