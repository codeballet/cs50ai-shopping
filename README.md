# Shopping introduction

Shopping is an Artificial Intelligence that predicts whether online shopping customers will complete a purchase or not.
When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session.
It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn’t planning to complete the purchase. How could a website determine a user’s purchasing intent? That’s where machine learning comes in.

## Method

The machine learning algorithm for this AI trains a "nearest-neighbor classifier" that predicts two values:

- Sensitivity, or the true positive rate, which is the proportion of examples in the testing data correctly identified as ending with a purchase.
- Specificity, or the true negative rate, which is the proportion of examples in the testing data correctly identified as not ending with a purchase.

# Using the AI

You may experiment using the AI with the provided `shopping.csv` file, containing training and testing data from about 12,000 online user sessions.

To run the AI with the provided data, you may enter:

```
python shopping.py shopping.csv
```

For generating your own data, please see the provided csv file for the data structure.

## Dependencies

The AI uses the `scikit-learn` library, which may be installed running the command:

```
pip install -r requirements.txt
```

Or:

```
pip install scikit-learn
```

# Intellectual Property Rights

MIT

The data in the `shopping.csv` file was provided by [Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007/s00521-018-3523-0)
