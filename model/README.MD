## File structure
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
Run this file to predict the helpfullness label for a dataset. Make sure to specify the required arguments:\
`python classify.py --output_file file.txt --dataset_dir ./review.csv --model_dir model.MODEL`
1. `output_file`: the text file directory which will contain the predictions. The predictions will conisist of
a set of booleans seperated by a newline.
2. `dataset_dir`: the directory where the dataset for which the predictions have to be made can be found. Make
sure that the corresponding dataframe consists of a `review` column, as well as any other numerical columns. Notice that the additional columns have to be the same for the training phase, so the dataset architecture has to remain constant.
3. `model_dir`: directory where the trained model can be found.  

### utils.py
File containing general utility functions

### train_model.py
Run this file to train a new model. Make sure to specify the required arguments as such:\
`python train_model.py --output_dir output --dataset_dir ./review.csv`
1. `output_dir`: the directory in which the output will be stored. The output can be found in a folder named as the current date and time in the specified output folder
2. `dataset_dir`: the directory of the dataset on which to train the model. Make sure that the dataset contains at least a `reviews` column, a `label` column. Extra (**numerical**) columns can also be added to be included in the final neural network.