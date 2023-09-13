#!/usr/bin/env python
# coding: utf-8

# In[20]:


from textblob import TextBlob


# In[21]:


from textblob.sentiments import NaiveBayesAnalyzer
blob = TextBlob("Enter your text here", analyzer=NaiveBayesAnalyzer())
blob.sentiment


# In[22]:


Text = TextBlob("Google brain is unique and unparallelebla. "
                "Robotics is fast moving industry ")
Text.words
Text.sentences


# In[23]:


from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor ()
blob = TextBlob("Machine Learning Coding is best thing in the world",np_extractor=extractor)
blob.noun_phrases


# In[24]:


blob = TextBlob("Machine Learning Coding is best thing in the world")
blob.tags


# In[25]:


blob = TextBlob("Machine Learning Coding is best thing in the world")
blob.ngrams(n=3)


# In[27]:


import csv
import os
from textblob import TextBlob

folder_path = "./sentiment_analysis"
output_file = "sentiment_output.csv"

with open(output_file, "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Line", "Polarity", "Subjectivity", "Label"])
    
    for subfolder_name in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(sub_path):
            print("Analyzing subfolder:", subfolder_name)
            for file_name in os.listdir(sub_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(sub_path, file_name)
                with open(file_path, "r") as file:
                    for line in file:
                        blob = TextBlob(line)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity

                        label = ""
                        if polarity > 0:
                            label = "Pos"
                        elif polarity < 0:
                            label = "Neg"
                        else:
                            label = "Neutral"

                        writer.writerow([line.strip(), polarity, subjectivity, label])


# In[ ]:




