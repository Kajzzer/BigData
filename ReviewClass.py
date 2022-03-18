from data_cleaning import word_tokenizer

import spacy

class Review():

    fr_core_news_sm = spacy.load("de_core_news_sm")
    de_core_news_sm = spacy.load("de_core_news_sm")
    en_core_web_sm = spacy.load("en_core_web_sm")

    def __init__(
        self, review_id, product_id, product_parent, product_title, vine, 
        verified_purchase, review_headline, review_body, review_date,
        marketplace_id, product_category_id, label
        ):
        self.review_id = review_id
        self.product_id = product_id
        self.product_parent = product_parent
        self.product_title = product_title
        self.vine = vine
        self.verified_purchase = verified_purchase
        self.review_headline = review_headline
        self.review_body = review_body
        self.review_date = review_date
        self.marketplace_id = marketplace_id
        self.product_category_id = product_category_id
        self.label = label
        
    def SelectLanguageModel(self):
        if self.marketplace_id == 2:
            self.model = Review.fr_core_news_sm
        elif self.marketplace_id == 3:
            self.model = Review.de_core_news_sm
        else:
            self.model = Review.en_core_web_sm

    def ReviewBodyCharCount(self):
        return len(self.review_body)

    def ReviewBodyWordCount(self):
        return len(word_tokenizer(self.review_body))

    def TaggedReviewBody(self):
        if not hasattr(self, 'model'):
            self.SelectLanguageModel()

        sentence = self.model(self.review_body)
        return [{'text': word.text, 'pos': word.pos_, 'tag': word.tag_} for word in sentence]
