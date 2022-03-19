### File structure
1. transformer.py
2. parameters.py
3. classify.py
4. utils.py
5. train_model.py

### transformer.py
Contains the transformer model architecture

### parameters.py
Centrally contains all the model parameters

### classify.py
Run this file to predict the helpfullness label for a dataset. Make sure to specify the required arguments:
`python classify.py --output_file file.txt --dataset_dir ./review.csv --model_dir model.MODEL`