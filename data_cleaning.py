# import nltk
# nltk.download('wordnet')
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re


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

def fix_title_headline(dataframe, dataframe_row):
    """
    Fix the product title and review headline if they have been mixed up
    """
    # get the possible titles of the product id of this row
    rows = dataframe[dataframe['product_id'] == dataframe_row['product_id']]

    # the unique titles of this product
    unique_titles = rows['product_title'].unique()

    # get the review headline
    review_headline = dataframe_row['review_headline']

    # check if the review_headline is in the unique titles
    if review_headline in unique_titles:

        # replace the review headline of the row with the title
        dataframe_row['review_headline'] = dataframe_row['product_title']

        # replace the title of the row with the review headline
        dataframe_row['product_title'] = review_headline

    # return the modified row
    return dataframe_row

def remove_accent_spam(text, threshold=0.075):
    """
    Remove the accents if they have been spammed in the text.
    If there is an accent spam in the text, that text only includes
    the vowels with acute accents instead of the i.
    """
    # the total number of characters
    total = len(text)

    # the number of accents
    acutes = 0

    # the map of acute char with their non accented variant
    acute_map = np.array([['á', 'a'], ['Á', 'A'], ['é', 'e'],
                          ['É', 'E'], ['ớ', 'o'], ['ó', 'o'],
                          ['Ó', 'O'], ['ú', 'u'], ['Ú', 'U']])

    # iterate over the characters and check if acute is in their unicode name
    for char in text:
        if char in acute_map[:, 1]:
            return text

    # clean the text
    for [accent, char] in acute_map:
        text = re.sub(accent, char, text)

    # return the cleaned text
    return text

# this can be used to ensure that we have a space after a sentence stopper or after a comma
def fix_sents_replacement(matchobj):
    if matchobj.group(1) and matchobj.group(2): return matchobj.group(1) + ' ' + matchobj.group(2)
    if matchobj.group(3) and matchobj.group(4): return matchobj.group(3) + ' ' + matchobj.group(4)
    return matchobj.group(0)

# fix the sentence ends for the nltk sent tokenizer
def fix_sents(text):
    return re.sub("([.?!,])([A-Z])|(,)([a-z])", fix_sents_replacement, text)
