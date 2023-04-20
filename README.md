# COSI 114a Final Project Write Up - Timothy Obiso

## Introduction

For my final project, I wanted to implement a multi-class classifier. [I like to read](https://www.goodreads.com/user/show/23932517-timothy-obiso) and wanted to explore how well some of the models we explored this semester would perform on a task related to reading. Based on this, I decided to investigate how well a model could classify the genre of a book based on its summary.  Next, I explored different publicly available datasets that contained the data I needed. Some datasets I rejected did not have enough data points (books), had too many or too few genres, or did not have long enough summaries. The dataset I trained this model does have problems, however, many of these were dealt with in pre-processing or others exist in every dataset available for this task. Another reason this is interesting to me is because some of the books I have read are in this dataset!

## Data

### The Dataset
For this project, the dataset I am using is called [Book Genre Prediction](https://www.kaggle.com/datasets/athu1105/book-genre-prediction) and is available on Kaggle. This dataset is comprised of three columns, book title, summary, and genre. The dataset does not specify how the summaries were obtained or how these books were chosen (only that this is a larger dataset based off of a previous one), but it does say that the genres were human-approved/human-annotated. In total, there are 4542 books in this dataset. There is no specified train/dev/test set, leaving me to split it myself in code. I made the call not to have the same test/train/dev sets each time and chose to separate them the using the same ratios, but not the exact same split. One of the biggest reasons for this is that the dataset is not that large to begin with. Another is that it is fairly unbalanced. There are three genres that only have 100 books each (sports, travel, psychology) and one that has 111 data points (romance). The books that appear next most frequently appear four times as often (crime), and the most frequent genre (thriller) appears about twice as often as crime does.

Based on initial testing, the performance of this model is moderately influenced (within a few percentage points) by the split into train/dev/test. If the data were randomly separated into standard train/test/dev sets to be reused, I would not be able to accurately speak on the performance of the model and would be speaking to the fit of the specific data split. In an ideal world, this database would be much larger and the influence of the split would not be so great; however, the values reported are averages to accurately discuss how well these models performed at tagging a genre based on the summary.

I split the data myself into train/dev/test in code through two calls to `train_test_split`. 
```
X = cv.fit_transform(database['summary'])  
y = database['genre'].values  
X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)  
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
```
where `cv` is either a CountVectorizer or a FeatureUnion. `stratify=y` is set because this dataset is unbalanced and this would ensure an equal ratio of each label is in each of the training, dev, and test sets.

There are ten genres (classes) in total (N=train/dev/test)
1. thriller (N=818/103/102)
2. fantasy (N=701/87/88)
3. science (N=517/65/65)
4. horror (N=480/60/60)
5. history (N=480/60/60)
6. crime (N=400/50/50)
7. romance (N=89/11/11)
8. sports (N=80/10/10)
9. travel (N=80/10/10)
10. psychology (N=80/10/10)

Here are some abbreviated examples from the dataset

|ID| Title | Summary | Genre |
|--|-------|----------|-------|
|2508|Man on Fire| In Italy, wealthy families often hire bodyguards to protect family members from the threat of kidnapping. When Rika Balletto urges her husband Ettore, a wealthy textiles producer living in Milan, to hire a bodyguard for their daughter Pinta, he is doubtful but agrees. After some searching, he finally settles for an American named Creasy. Creasy, once purposeful and lethal, has become a burnt-out alcoholic. To keep him occupied, his companion Guido suggests that Creasy should get a job, and offers him to set him up as a bodyguard; thus he is being hired by the Ballettos, where he meets his charge, Pinta. Creasy... (517 words)|**thriller**|
|4296|The Final Empire| What if the whole world were a dead, blasted wasteland? Mistborn For a thousand years the ash fell and no flowers bloomed. For a thousand years the Skaa slaved in misery and lived in fear. For a thousand years the Lord Ruler, the "Sliver of Infinity," reigned with absolute power and ultimate terror, divinely invincible. Then, when hope was so long lost that not even its memory remained, a terribly scarred, heart-broken half-Skaa rediscovered it in the depths of the Lord Ruler's most hellish prison... (303 words)  | **fantasy**|


### Preprocessing
The data goes through a some preprocessing before it is tokenized. By looking at the actual data, I noticed that there were many summaries that ended with `(less)`. This is a text button that alternates between `(less)` and `(more)` and I recognize it from the legacy Goodreads book summary page. While this tag does tells me where the summaries (or at least some of them) came from, it is not going to be helpful when determining genres, so this string was removed before tokenization occured. Another type of preprocessing that I performed was removing punctuation from the summaries as well. After this was completed, the data was tokenized and used to train the following models with the following results. 

## Development Set Results

### Model Selection
|Test|Model|Accuracy|
| -|-|-|
|1| Random Forest Unigrams | **57.82%**
2|Random Forest 1000 Unigrams |54.01%
3|Random Forest 1000 Unigrams and Bigrams|54.94%
|
4|SVM 1000 Unigrams. Stop Words Removed. C = 1.0|**51.21%**
5|SVM 1000 Unigrams. Stop Words Removed C = 0.5 |42.90%
6|SVM 1000 Unigrams. Stop Words Removed C = 0.1 |25.39%
|
7|Naive Bayes (BernoulliNB) Stop Words Removed. Unigrams |51.93%
8|Naive Bayes (BernoulliNB) Unigrams |52.58%
9|Naive Bayes (BernoulliNB) Stop Words Removed. Unigrams and Bigrams |50.85%
10|Naive Bayes (BernoulliNB) Unigrams and Bigrams |45.71% 
11|Naive Bayes (BernoulliNB) Stop Words Removed. Bigrams |43.56%
12|Naive Bayes (BernoulliNB) Bigrams |41.42% 
13|Naive Bayes (ComplementNB) Stop Words Removed. Unigrams |65.31%
14|Naive Bayes (ComplementNB)  Unigrams |62.45%
15|Naive Bayes (ComplementNB) Stop Words Removed. Unigrams and Bigrams |62.88%
16|Naive Bayes (ComplementNB)  Unigrams and Bigrams |65.23% 
17|Naive Bayes (ComplementNB) Stop Words Removed. Bigrams |53.86%
18|Naive Bayes (ComplementNB)  Bigrams |58.37% 
19|Naive Bayes (MultinomialNB) Stop Words Removed. Unigrams |**65.56%**
20|Naive Bayes (MultinomialNB) Unigrams |65.14%
21|Naive Bayes (MultinomialNB) Stop Words Removed. Unigrams and Bigrams |64.24%
22|Naive Bayes (MultinomialNB) Unigrams and Bigrams |63.52% 
23|Naive Bayes (MultinomialNB) Stop Words Removed. Bigrams |52.58%
24|Naive Bayes (MultinomialNB)  Bigrams |50.43% 
25|Naive Bayes (GaussianNB) Stop Words Removed. Unigrams |46.14%
26|Naive Bayes (GaussianNB) Unigrams |40.99%
27|Naive Bayes (GaussianNB) Stop Words Removed. Unigrams and Bigrams |44.42%
28|Naive Bayes (GaussianNB) Unigrams and Bigrams |46.78% 
29|Naive Bayes (GaussianNB) Stop Words Removed. Bigrams |44.46%
30|Naive Bayes (GaussianNB)  Bigrams |39.48%


### Hyperparameter Tuning

Test|Model|$\alpha$|Accuracy|
|-|-|-|-|
|31|MNB|1.0|**65.56%**
|32|MNB|0.5|64.88%
|33|MNB|0.1|64.04%

## Test Set Results

### Feature Testing
(stop words removed in all features, $\alpha=1$)
|Model|Specifics|Accuracy|
| -|-|-|
|MNB (default/v0)|All Unigrams|65.14%|
|MNB (v1)|6000 Unigrams, 1000 Bigrams|67.28%
|MNB (v2)|6000U, 1000B, 10,000 whitespaced character 4-grams|68.23%
|MNB (v3)|6000U, 1000B, 10,000 character 4-grams, 300 *rare* unigrams (max_df=.3)|**69.31%**

### Classification Report (v3)
|genre|precision|recall|f1 score|support
|-|-|-|-|-|
|crime|0.77|0.72|0.74|50
|fantasy|0.79|0.69|0.74|87
|history|0.70|0.63|0.67|60
|horror|0.70|0.53|0.60|60
|psychology|0.69|0.90|0.78|10
|romance|0.33|0.55|0.41|11
|science|0.62|0.74|0.68|65
|sports|0.80|0.80|0.80|10
|thriller|0.68|0.76|0.72|103
|travel|0.73|0.80|0.76|10


|metrics|precision|recall|f1 score|support
|-|-|-|-|-|-|
|**accuracy**|||**0.69**|466
|macro-average|0.68|0.71|0.69|466
|weighted average| 0.70|0.69|0.69|466

## Discussion

I assumed **MultinomialNB** or **ComplementNB** would perform best out of all the Naive Bayes configurations because these handle **multiclass classification** and **unbalanced data** especially well, respectively. My suspicions were correct as these were the two best of the Naive Bayes models. However, I was not expecting these to perform significantly better than Random Forest and SVM (as well as other models not shown here such as Logistic and Linear Regression and Decision Tree).

I thought bigrams (or unigrams and bigrams) would be a slight step up from just unigrams in almost all cases, but the opposite proved to be true. Considering only bigrams caused the model to perform worse. Bigrams must be combined with unigrams or other features in order to be effective. Using more than 1000 bigrams exhibited a significant drop in performance. This demonstrates that they can be *an* important feature, yet they are not the *most* important feature.

The inclusion of the "rare" unigrams and character 4-grams gave large boosts in accuracy to categories like romance and travel and smaller boosts in accuracy to most others. I was not surprised that the "rare" unigrams were helpful on categories not seen often in training. I was surprised that character 4-grams were. I account this improvement to the fact that these character 4-grams are some sort of pseudo-lemmatization/stemming.

I believe my model performed well. As I mentioned above, I also ran the moels many times to ensure the values presented were accurate and "average" values. I have run the models 20 times each. Within those 20 times, the lowest accuracy I saw for *v3* was 66% and the highest was 73%. The average accuracy was approximately 69.58%. On Kaggle, the highest accuracy I saw that did not use things off-limits in this project was 69% (as reported). That model was also MNB. Since my model frequently got above 69%, I feel it works very well overall.

## Conclusion

This model is able to accuractely predict the genre of a book based on its summary  69-70% of the time. From the baseline score of 65% using the same model, this is a considerable improvement from hyperparameter tuning and feature combinations.

It was more of a challenge to increase the accuracy from the baseline accuracy than I expected. I was glad I was able to find a combination of features and hyperparameters that worked well, but it took a lot of trial and error to do so. I likely tried over 100 feature combinations, *v3* reliably worked the best out of all of them.

If I could spend more time on this problem, I would build my own much **larger** and **balanced** corpus to do so. I would keep the same classes of categories as I feel they are distinct enough to be separable by a human with little trouble, but aim to have up to 50,000 books with an even genre split (5,000-10,000 human tagged, the rest tagged from the first tag found on Goodreads). I would also separate the data myself into a train/test/dev split **in advance**. Finally, I would look to use nltk, other stop word sets, BERT and other transformer models, or neural networks to accomplish this task.
