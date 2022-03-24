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
        if self.marketplace_id == 2:
            model = Review.fr_core_news_sm
        elif self.marketplace_id == 3:
            model = Review.de_core_news_sm
        else:
            model = Review.en_core_web_sm
            
        sentence = model(self.review_body)

        length = self.ReviewBodyWordCount()

        if length > 1:

            self.pos_verb_ratio = sum(word.pos_ == "VERB" for word in sentence) / length
            self.pos_propn_ratio = sum(word.pos_ == "PROPN" for word in sentence) / length
            self.pos_aux_ratio = sum(word.pos_ == "AUX" for word in sentence) / length
            self.pos_adp_ratio = sum(word.pos_ == "ADP" for word in sentence) / length
            self.pos_noun_ratio = sum(word.pos_ == "NOUN" for word in sentence) / length
            self.pos_num_ratio = sum(word.pos_ == "NUM" for word in sentence) / length
        else:
            self.pos_verb_ratio = 0
            self.pos_propn_ratio = 0
            self.pos_aux_ratio = 0
            self.pos_adp_ratio = 0
            self.pos_noun_ratio = 0
            self.pos_num_ratio = 0