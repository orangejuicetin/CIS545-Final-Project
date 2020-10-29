#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Dataset Project
# ## Justin Choi
# 
# Hi there! For this project, I opted to use the COVID-19 research challenge dataset, which they named "CORD-19" (I promise, the title wasn't a typo haha). The dataset of articles was created by the Allen Institute for AI, Chan Zuckerberg Initiative, Microsoft Research, NIH, and more; if you wanna check it out for yourself, you can either [download](https://www.semanticscholar.org/cord19/get-started "AI2 link to dataset") it or check out the [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=568 "kaggle page!") webpage for it!

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import json


# # The Data 
# Thanks to all the ~ useful ~ skills we've picked up over the course of this semester, we'll start with everyone's favorite tedious, time-consuming task - data cleaning! woo hoooooOOOOooOo who doesn't lüv missing values and weird formatting. 
# 
# First, we'll be utilizing the built-in `json`, `os`, and `glob` modules from Python to get each of the files from our directory and then extract the right text! From there, we'll be using our best friend `pandas` in order to aggregate all this text data into one dataframe along with it's associated metadata from the `metadata.csv`:

# In[2]:


metadata_df = pd.read_csv('./CORD-19-research-challenge/metadata.csv')

# Import all the json files
cord_19_folder = './CORD-19-research-challenge/'
list_of_files = []; # only going to take those from pdf_json! not pmc_json
for root, dirs, files in os.walk(cord_19_folder):
    for name in files:
        if (name.endswith('.json')):
            full_path = os.path.join(root, name)
            list_of_files.append(full_path)
sorted(list_of_files)
print('done')

# ALTERNATE

# all_json = glob.glob(f'{cord_19_folder}/**/*.json', recursive=True)
# len(all_json)


# In[ ]:


class JsonReader:
    def __init__(self, file_path):
        with open(file_path) as file: 
            content = json.load(file)
            # start to insert body text 
            self.paper_id = content['paper_id']
            self.body_text = [] 
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.body_text[:500]}...'

random_json = list_of_files[47404]
sample_article = JsonReader(random_json)
print(sample_article)


# Now that we've extracted all the necessary text data, let's load it onto a dataframe and clean it out so that we don't have any `null` values in important places (e.g. `title`, `body_text`, etc)

# In[4]:


input = {'paper_id': [], 'doi':[],  'title': [], 'abstract': [], 'body_text': [], 'authors': [], 'journal': []}

for i, entry in enumerate(list_of_files):
    if i % (len(list_of_files) // 25) ==    0:
        print(f'Processing {i} of {len(list_of_files)}')
    try: 
        article = JsonReader(entry)
    except Exception as e: 
        continue #means that we don't have a valid file format  
    
    metadata = metadata_df.loc[metadata_df['sha'] == article.paper_id]
    if len(metadata) == 0:
        continue # no such metadata for paper in our csv, skip

    input['body_text'].append(article.body_text)
    input['paper_id'].append(article.paper_id)

    # add in metadata 
    title = metadata['title'].values[0] 
    doi = metadata['doi'].values[0] 
    abstract = metadata['abstract'].values[0] 
    authors = metadata['authors'].values[0] 
    journal = metadata['journal'].values[0] 

    input['title'].append(title)
    input['doi'].append(doi)
    input['abstract'].append(abstract)
    input['authors'].append(authors)
    input['journal'].append(journal)


# In[5]:


covid_df = pd.DataFrame(input, columns=['paper_id', 'doi', 'title', 'abstract', 'body_text', 'authors', 'journal'])
print('finished creating dataframe from input dictionary')
rows, cols = covid_df.shape
print(f'number of rows: {rows}')
covid_df.info()


# In[6]:


covid_df.dropna(inplace=True)
print('finished dropping articles with null abstracts/body text/titles')
rows, cols = covid_df.shape
print(f'number of rows: {rows}')


# In[7]:


covid_df['body_word_count'] = covid_df['body_text'].apply(lambda x : len(x.strip().split()))
covid_df['body_unique_count'] = covid_df['body_text'].apply(lambda x : len(set(x.strip().split())))


# In[8]:


# visualiation check to see if data is finished being cleaned 
covid_df.head()


# In[9]:


covid_df.info()


# In[10]:


#check to see if there are duplicates
covid_df['abstract'].describe()


# In[11]:


covid_df.drop_duplicates(subset=['abstract', 'body_text'], inplace=True)


# In[12]:


covid_df.describe()


# Yayyy go cleaned data! Now that we have a cleaned out dataframe, one of the first things we want to do is find the language of each of these research articles, as not all of them are in English! We'll do some EDA on this in a bit, but for the rest of the project we're going to filter out any articles that aren't in English, just so we can simplify our modeling later down the line: 

# In[13]:


from langdetect import detect
from tqdm import tqdm

languages = [] # make list that you can port directly into covid_df as column

for i in tqdm(range(len(covid_df))):
    row_text = covid_df.iloc[i]['body_text'].split(" ") 
    lang = 'en' # set default lang to be english 

    # try to just use the intro 25 words to detect language
    try:
        if (len(row_text)) > 125: 
            lang = detect(" ".join(row_text[:125]))
        elif(len(row_text)) > 0:
            lang = detect(" ".join(row_text))
    except Exception as e: # if body doesn't work, let's try abstract
        try: 
            lang = detect(covid_df.iloc(i)['abstract'].split())
        except Exception as e:
            lang = 'dunno'
            continue
    finally:
        languages.append(lang)
        


# In[14]:


lang_array = np.asarray(languages)
covid_df['language'] = lang_array
lang_dict = {}
for language in lang_array:
    if language in lang_dict:
        lang_dict[language] += 1
    else: 
        lang_dict[language] = 1
lang_dict


# In[15]:


# test to see if languages were detected correctly
covid_df[covid_df['language'] == 'nl'].head(10)


# In[16]:


which_language = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French', 
    'it': 'Italian', 
    'zh-cn': 'Chinese', 
    'pl': 'Polish', 
    'cy': 'Welsh', 
    'de': 'German', 
    'pt': 'Portugese', 
    'nl': 'Dutch', 
    'dunno': 'Unknown'
}


# In[17]:


test = covid_df['language'].apply(lambda x : which_language[x])
covid_df['language'] = test
covid_df['language']


# # EDA 
# 
# Now, we can start exploring the cleaned dataset! Since we just extracted the languages of each of the text, let's just see how they're distributed as a warm-up to our visualizations:

# In[18]:


import seaborn as sns


# ## Language Distribution
# For the rest of the visualization throughout this project, I'll be using Seaborn over matplotlib thanks to its additional nice features as well as ~_aesthetic_~ appeal, so if you have anything you want to refer to, you can always check the [documentation](https://seaborn.pydata.org/)!

# In[19]:


lang_distribution = covid_df['language'].value_counts()
lang_distribution


# (For this language plot, I changed to across logarithmic scale so that 
# languages other than English could actually show up on the bar plot lol) 

# In[21]:


sns.set()
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_yscale('log')
plt.xticks(rotation=45)
sns.barplot(x=lang_distribution.index, y=lang_distribution.values, ax=ax, palette=sns.color_palette('Blues_r', len(lang_distribution)))
plt.show()


# In[22]:


covid_df[covid_df['language'] == 'French']


# ## Journal Contributions
# Next up, let's see what journal has published the most relevant research on COVID-19 and coronaviruses within this specific dataset!
# 
# Since we can only fit so many journals onto one bar plot, we're going to limit the journals we display to those that have contributed more than 100 articles to this dataset:

# In[23]:


journal_dist = covid_df['journal'].value_counts()
above_100 = journal_dist[journal_dist.values > 100]
above_100


# Now that we have a filtered list, let's organize it from greatest to least to see how different the contributions are from various journals:

# In[24]:


fig, ax_sized = plt.subplots(figsize=(20, 20))
plt.xticks(rotation=90)
sns.barplot(x=above_100.index, y=above_100.values, ax=ax_sized, palette=sns.color_palette('Blues_r', len(above_100)))
plt.show()


# ## Publication Date
# Let's now analyze when our research papers were published! Given that COVID-19 is only a very recent species of the many other coronaviruses we know of, it'll be interesting to see how research activity has developed over the course of time, and if intuition serves us right, we'll presumably see a large spike in research publication from the past few months.

# First, we're going to change the `publish_time` column into `datetime` objects so we can better work with the data! 

# In[25]:


from datetime import datetime

test = pd.merge(covid_df, metadata_df[['sha', 'publish_time']], left_on='paper_id', right_on='sha', how='left')
test['publish_time'] = pd.to_datetime(test['publish_time'], infer_datetime_format=True)
test


# In[26]:


covid_df = test
covid_df.head() 


# Now that we have the data in the right format, let's extract the publication year and visualize it!

# In[27]:


years_df = covid_df.copy()
years_df['publish_year'] = covid_df['publish_time'].apply(lambda x : x.year)
dates = years_df['publish_year'].value_counts()
dates


# In[28]:


fig, ax = plt.subplots(figsize=(20, 20))
plt.xticks(rotation=45)
sns.barplot(x=dates.index, y=dates.values, ax=ax, palette=sns.color_palette('Blues', len(dates)))
plt.show()


# A super interesting thing to note here is that coronavirus research was relatively low level up until a huge spike in 2004; this __ _directly coincides_ __ with the SARS outbreak from 2002-2003, hence why we can observe a huge spike in research published the following year (!!) This was only further fueled after the outbreak of MERS in 2012, a different species of coronavirus that began to spread in the Middle East.    
# 
# Another thing to note is that 2020 is only around halfway through at the time of this writing (May, 2020) - yet, enough research has already been published (and of course, countless other scientsts are likely working on new research as we speak) that it's on pace to likely crush numbers from any prior years. 

# ### Let's now focus on the past two years of research, and compare month by month to see how research activity has developed. Again, if our assumptions are correct, we should see a large spike in activity in the first few months of 2020: 

# In[29]:


two_years_df = years_df.copy()
two_years_df['month'] = years_df['publish_time'].apply(lambda x : x.month)
nineteen_df = two_years_df[two_years_df['publish_year'] == 2019]
twenty_df = two_years_df[two_years_df['publish_year'] == 2020]

nineteen_counts = nineteen_df['month'].value_counts()
nineteen_counts.sort_index(inplace=True)
nineteen_counts.rename('2019', inplace=True)
nineteen_counts.fillna(value=0, inplace=True)


twenty_counts = twenty_df['month'].value_counts()
twenty_counts.sort_index(inplace=True)
twenty_counts.rename('2020', inplace=True)
twenty_counts.fillna(value=0, inplace=True)
twenty_counts.iloc[5:12] = 0

combined_counts = pd.concat([nineteen_counts, twenty_counts], axis=1)
combined_counts.reset_index(inplace=True)
number_to_month = {
    1: 'Jan', 
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sept',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}
combined_counts['month'] = combined_counts['index'].apply(lambda x : number_to_month[x])
combined_counts.drop(['index'], axis=1, inplace=True)
month_data = combined_counts.melt('month', var_name='Year', value_name='Number of Papers')

fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x='month', y='Number of Papers', hue='Year', ax=ax, data=month_data, palette='coolwarm')
plt.show()


# ## Interesting Note
# The 2020 months are actually supposed to have even _more_ articles attributed to them, but the `metadata_df` unfortunately had the wrong publication date for quite a few of the 2020 articles, (i.e. it defaulted to just setting them as being published in December when they were just in a journal published in, say, June 2020, which is in the "future"); because of this, I had to remove around ~ 150 articles from the 2020 dataset.
# 
# Even **_then_** we can see that 2020 has obviously had a massive spike in research. Again, this has some cool correspondences to the actual timeline of the disease: late-December/early-January is when WHO first made a risk assessment of the disease and China publicly shared COVID-19's genetic sequence; late-January WHO declared a global health emergency, and February was Wuhan went under lockdown, the disease began spreading rapidly to Europe and the U.S., Wuhan went under lockdown, and the US confirmed it's first case, alongside other nations such as South Korea and France who had also reported new cases. And this directly reflects in our data, as it's evident that March had a __MASSIVE__ spike in research, with pretty much double the amount of papers published. 
# 

# # NLP + Feature Extraction
# Now that we've successfully cleaned out the data and gotten all the text in a consistent fashion, now we're going to create a bag-of-words model and vectorize each of the documents! This'll then allow us to do some better visualization and run some cool ~ _machine learning_ ~ like PCA and t-SNE to both reduce dimensionality and visualize this better! Our main tool to do this will be NLTK, so if there are any questions concerning any of the methods used, just check out their documentation [here](https://www.nltk.org/ "NLTK"):

# In[30]:


dropped = covid_df[covid_df['language'] == 'English'] # i.e. only select articles written in english, as it'll help parsing/NLP 


# In[31]:


covid_df['language'].describe()


# First, let's import NLTK and download the necessary pre-trained model as well as stopwords to filter out our text!

# In[32]:


# NLP analysis using NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
import nltk.tokenize as t
import re

stop_words = list(set(stopwords.words('english')))
print(stop_words)


# Since biomedical journals also have a ton of reoccuring words that are common-place, we'll add these to our list of stopwords so that we can further remove useless data and ultimately give us a cleaner model at the end:

# In[33]:


# add in additional stopwords frequently used in biomedical/research articles
bio_stop_words = ['doi', 'preprint', 'copyright', 'www', 'PMC', 'pmc', 'al.', 'fig', 'fig.', 'permission', 'used', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights', 'reserved', 'biorxiv', 'medrxiv', 'license', 'CZI', 'czi']

for word in bio_stop_words:
    if word not in stop_words:
        stop_words.append(word)
        
print(stop_words)


# In[34]:


# helper function to remove punctuation from sentences 
def remove_punctuation(sentence):
    sentence = re.sub(r'[^\w\s]','', sentence)
    return sentence

def remove_stopwords(sentence):
    return [word for word in sentence if word not in stop_words]

def parse_text(text): 
    sentences = t.sent_tokenize(text)
    sentences = [sentence.lower() for sentence in sentences] # lower case the text so that we can correctly remove text
    cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences]
    tokenized_sentences = [t.word_tokenize(clean_sentence) for clean_sentence in cleaned_sentences]
    filtered_sentences = [remove_stopwords(t_sentence) for t_sentence in tokenized_sentences]
    tokens = ''
    for sentence in filtered_sentences:
        sentence_str = ' '.join(sentence)
        tokens = tokens + sentence_str + ' '
    return tokens


# In[35]:


# checking to see if the function parse_text works correctly

test_article = covid_df.iloc[2020]['body_text']
test_article
output = parse_text(test_article)
print(output)


# In[36]:


tqdm.pandas()
test = covid_df.copy()
test['parsed_text'] = test['body_text'].progress_apply(parse_text)
test


# In[37]:


# assign the copy we made with the parsed text to our current working dataframe
covid_df = test


# ## Word Count Distribution
# Since we're already working with text data here, an interesting thing to note here is how our journals are distributed in terms of length! We get a good idea of how long our papers are here, for the most part, our journals are under 5500 words in length, and it's only a very small part of our dataset that has extremely long articles (e.g. the max one with 172000 words in total, sheesh). __Note: this word count data was before we cleaned it using NLTK, so this includes all stopwords as well.__

# In[38]:


sns.set(style='white', palette='dark', color_codes=True)


# In[39]:


fig, ax = plt.subplots(figsize=(20, 20))
plt.ylabel('Percentage of Articles', fontsize=15)
sns.distplot(covid_df['body_word_count'], ax=ax, color='g')
plt.title('Total Word Count', fontsize=25)
plt.xlabel('Words', fontsize=15)
plt.show()
covid_df['body_word_count'].describe()


# In[40]:


fig, ax = plt.subplots(figsize=(20, 20))
plt.ylabel('Percentage of Articles', fontsize=15)
sns.distplot(covid_df['body_unique_count'], ax=ax, color='m')
plt.title('Unique Word Count', fontsize=25)
plt.xlabel('Words', fontsize=15)
plt.show()
covid_df['body_unique_count'].describe()


# ## Vectorizing Our Documents!
# Now comes the NLP part of our project - we'll be utilizing the idea of tf-idf (term frequency - inverse document frequency) in order to make each of our documents into a workable, normalized vector that we can manipulate and compare with other document vectors! We'll be using scikit's inbuilt feature extraction package that has a vectorizer for tf-idf for this task ~ 
# 
# > __Note__: I limit the max number of features allowed in the vectorizer to 2<sup>12</sup> because we want to make sure this step doesn't take absolutely forever, but if we wanted to increase the model complexity to see if we could extract more useful information out of it, we can always change this in the future!
# > 
# 
# Again, if there are any questions on this part, the documentation for the tf-idf vectorizer can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.fit_transform "scikit documentation")

# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=2**12)

# get all of the text that was processed by our NLTK parser
all_text = covid_df['parsed_text'].values
tfidf_matrix = tfidf.fit_transform(all_text)
tfidf_matrix.shape


# ## mAchiNe LEaRnINg !!
# Now comes the ~_fancy_~ machine learning algoritms; for this dataset, we're going to mainly use two unsupervised learning methods (since there's nothing to really classify/regress per se for this dataset of articles) - Latent Semantic Analysis (LSA) and K-Means Clustering!
# 
# LSA is actually very similar to PCA in the sense that it's great for dimensionality reduction, but it focuses specifically on the tf-idf matrix we've made, so we'll be using the `TruncatedSVD` package in scikit in order to perform it (since our term-document matrix is far more sparse than the covariance matrix a typical PCA would run on). This'll help us map these matrices to "semantic spaces" that have lower dimensionality. 
# 

# In[42]:


from sklearn.decomposition import TruncatedSVD
t_svd = TruncatedSVD(n_components=100, random_state=2022)
tfidf_reduced = t_svd.fit_transform(tfidf_matrix)
tfidf_reduced.shape


# ## K-Means Clustering
# Now is where we'll be running our clustering algorithm on the reduced dimensionality data! This'll give us a natural partitioning of our data into a number of clusters, and we'll find the right range of `k`-values using the "elbow method" (i.e. we'll find where the reduction in distortion begins to taper off given a value of `k`)

# In[43]:


from sklearn.cluster import KMeans
import scipy

# create an array of distortion values so we can visualize and find the elbow
distortion = []
K = range(3, 60)
for k in K:
    km = KMeans(n_clusters=k, random_state=2022)
    km.fit(tfidf_reduced)
    distortion.append(sum(np.min(scipy.spatial.distance.cdist(tfidf_reduced, km.cluster_centers_, 'euclidean'), axis=1))     / tfidf_reduced.shape[0])
    if k % 5 == 0:
        print(f'distortion for {k} clusters')


# In[44]:


sns.set(palette='dark')
fig, ax = plt.subplots(figsize=(20, 20))
sns.lineplot(x=K, y=distortion)
plt.title('Distortion Plot', fontsize=20)
plt.xlabel('# of Clusters "K"', fontsize=15)
plt.ylabel('Distortion', fontsize=15)
plt.show()


# Although it's unclear at first, we can see that the "staggering" effects begin to happen around 20 and we begin to get less and less noticable reductions around 30, so we'll set `k = 25` for our clustering. We'll run it again with this value and now add a column to our dataframe denoting the cluster assignments. 
# 

# In[45]:


# set number of clusters, k 
k = 25

# perform the algorithm with the given value of k 
km = KMeans(n_clusters=k, random_state=2022)
assignments = km.fit_predict(tfidf_reduced)


# In[46]:


covid_df['cluster_assigments'] = assignments
covid_df.head()


# ## t-SNE
# Since we want to be able to visualize our data and at least get somewhat of a sense of how our data is organized (beyond just assignment numbers and meaningless data tables), we're going to try and bring down the dimensionality of our data even further so that we can visualize it in 2D, hence plot it in our notebook, and what better way to do this than t-SNE! For the sake of computational efficiency, we're going to utilize the reduced dimension matrix of data from our LSA, as otherwise it's going to take forever, but if you have the time, you should check it out with the entire featured term-document matrix `tf-idf`!
# 
# __Side Note:__ The LSA that we performed earlier using `TruncatedSVD` actually has the ability to project our text data onto just two dimensions as well, effectively doing what t-SNE is doing here, but t-SNE is slightly more advanced and effective for this task thanks to advanced mathematical machinery, and for the sake of diversity of methodology, we'll just stuck with t-SNE for now! (however, if I have time later, I'll include another graphic comparing which visualization method is better!)

# In[47]:


from sklearn.manifold import TSNE

t_sne = TSNE(verbose=1, perplexity=50, random_state=2020)
tfidf_2_dim = t_sne.fit_transform(tfidf_reduced)


# In[58]:


tfidf_2_dim


# In[84]:


fig, ax = plt.subplots(figsize=(20, 20))
sns.scatterplot(x=tfidf_2_dim[:, 0], y=tfidf_2_dim[:, 1])
plt.title('Basic t-SNE', fontsize=20)
plt.show()


# Of course, this gives us the clusters, but it'd be far more interesting if we could see the distinction between clusters using the labels that we had from K-Means! So, let's do just that!

# In[86]:


fig, ax = plt.subplots(figsize=(20, 20))
sns.scatterplot(x=tfidf_2_dim[:, 0], y=tfidf_2_dim[:, 1], hue=assignments, legend='full', palette=sns.hls_palette(k, l=0.43, s=0.6))
plt.show()


# ## Significance
# Although there are a few outliers in terms of coloring, the super cool thing that we can observe in this graph is that, although they were done separately, the K-Means clustering algorithm and t-SNE algortihm both agreed on how to cluster the data, indicating to us that there indeed is some consistency in our data, and that there must be some shared uniform characteristics among these clusters. By extracting these characteristics, we could actually quantify what connects these research papers together and potentially help researchers/scientists better explore the current literature on COVID-19 by helping them discover additional research they never would have made a connection to otherwise! (since both t-SNE and K-Means cluster using far higher dimensional features than just common search words/tags that we typically utilize when search for articles of interest)
# 

# In[66]:


# section where we'll be doing LDA 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# LDA improves upon the tf-idf model to make more than just a clustering - can now generate topics for each bpaper
vectorizers = []
for i in range(k): 
    vectorizers.append(CountVectorizer(min_df=7, max_df=0.8, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))


# In[67]:


processed_data = []

for i, vectorizer in enumerate(vectorizers): 
    try:
        processed_data.append(vectorizer.fit_transform(covid_df.loc[covid_df['cluster_assigments'] == i, 'parsed_text']))
    except Exception:
        print('not enough points in cluster')
        processed_data.append(None)


# In[68]:


NUM_TOPICS = 22
lda_models = []
for i in range(k):
    model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online', verbose=False, random_state=2022)
    lda_models.append(model)


# In[69]:


lda_models[0]


# In[70]:


clusters_lda = []

for i, model in enumerate(lda_models):
    if (i % 5 == 0 or i == len(lda_models)): 
        print(f'Processing cluster #{i}')
    if processed_data[i] != None:
        clusters_lda.append(model.fit_transform(processed_data[i]))


# In[75]:


def get_topics(model, vectorizer, top_n = 3):
    curr = [] 
    all_keywords = []

    for i, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[j], topic[j]) for j in topic.argsort()[:-top_n - 1:-1]]
        for word in words: 
            if word[0] not in curr:
                all_keywords.append(word)
                curr.append(word[0])

    all_keywords.sort(key=lambda x: x[1])
    all_keywords.reverse
    return_words = []
    for word in all_keywords:
        return_words.append(word[0])

    return return_words


# In[76]:


all_topics = []

for i, model in enumerate(lda_models):
    if processed_data[i] != None: 
        all_topics.append(get_topics(model, vectorizers[i]))


# In[77]:


all_topics[0][:10]


# In[80]:


f = open('topics.txt', 'w')
count = 0

for topic_list in all_topics:
    if processed_data[count] != None:
        print(', '.join(topic_list) + '\n')
        f.write(', '.join(topic_list) )
    else: 
        f.write('Not enough instances \n')
        print(', '.join(topic_list) + '\n')
        f.write(', '.join(topics_list) + '\n')
    count += 1

f.close() 


# # Challenges/Obstacles
# Overall I found this project super cool! Having something to build from the groundup is definitely intimidating at first, but when you get into the groove of it and start becoming super productive and familiar with your data and what you want out of it, you start doing some really cool things, and it's awesome to see that progression. That being said however, some of the challenges were:
# 
# 1. It was very hard to read the dataset in the first place; trying to determine what all these random folders and filenames mean is _extremely_ confusing and intimidating at first, and tbh I almost switched my research topic within the first few days because I got so spooked by the data, haha. Glad I stuck with it though, got to do some really cool things with this dataset at the end of the day. 
# 
# 2. Trying to figure out how break down the data into digestible bits! Beyond just debugging the NLTK parser and trying to make the process efficient so it didn't take forever, it was also a challenge just figuring out _how_ to get the text data. The fact that everything was in `json` format in huge folders on my computer definitely didn't help either, and I guess it was a good way to force myself to learn what `glob` and `os` are used for in Python LOL - learning how to plot and visualize well in Python using `seaborn` was also definitely an accelerated learning process, but definitely feel super confident with data visualization now thanks to this project. 
# 
# 3. Finally, just making sure that I applied LSA/K-means correctly so that I could get a nice cluster-coloring was a challenge, since you have to make sure all the dimensions line up as well and that you have the right inputs and outputs for every function and module you're using, so that was definitely a headache at times. 
# 

# # Conclusion / Potential Next Steps
# 
# Although there was definitely a lot of analysis and modeling done throughout the course of this project, there are definitely a ton of ways to take it further! ~~My original plan was to run LDA (Latent Dirichlet Allocation) to extract the respective topics from each of the clusters, but finals szn (rip) took a heavy toll on the amount of time I had left for this project, so I didn't end up including it within the final project. However !! might do so in the future, so that's in the works~~ (__EDIT:__ Got around to finishing the part about LDA! You can see a pretty cool breakdown of the topics within the printed statement, and if you want a consolidated list it's available in the `topics.txt` file! Now might try to do some more stuff with Plotly/Dash)  - otherwise, hope you found the project interesting! Definitely was super interesting for me, and grateful that I had this opportunity to work with a super cool dataset relevant to a current ongoing pandemic (not many times you can say that your school work is relevant to your everyday life, haha); thanks for an awesome semester! 
