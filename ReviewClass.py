class Review():
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

    def ReviewBodyCharCount(self):
        return len(self.review_body)