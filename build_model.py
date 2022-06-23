import re
import nltk
import pandas as pd
import numpy as np
import scipy
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer,util
from gbq_connection import google_cloud_connection
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity


class ModelBuilder(object):

    def __init__(self):
        self.time_stamp = datetime.now().replace(tzinfo=timezone.utc)
        self.gcloud_con = google_cloud_connection()
        query = """SELECT
            request__c_organisation_notes__c,
            request__c_keywords__c
            FROM
            (SELECT
                    request__c.keywords__c  AS request__c_keywords__c,
                    request__c.organisation_notes__c  AS request__c_organisation_notes__c,
                    COALESCE(SUM(CASE when  request__c.awarded_amount__c   IS NULL AND  request__c.disposition__c   = 'Approved' THEN  request__c.original_awarded_amount__c
            ELSE  request__c.awarded_amount__c   END), 0) AS request__c_sum_awarded_amount
                FROM `analytics-240815.salesforce_output.Request__c_w_extension_forecasts`
             AS request__c
                WHERE (NOT (request__c.extension_forecast ) OR (request__c.extension_forecast ) IS NULL) AND (request__c.keywords__c ) IS NOT NULL AND (request__c.organisation_notes__c ) IS NOT NULL
                GROUP BY
                    1,
                    2
                HAVING request__c_sum_awarded_amount > 15000) AS t3
            ORDER BY
            1"""

        self.data = self.gcloud_con.download_df('',query=query)



    def remove_stop_words(self,document):
        '''
        Remove uninformative words such as the, is, because, etc from the corpus

        Parameters:
        ------------------------
        document: The corpus to operate on

        Returns
        -------
            Corpus without stop words
        '''
        # df = self.data
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        cleaned_document = " ".join([word.strip() for word in document.split() if word not in stop_words and len(word) > 2])
        return cleaned_document


    def preprocess_document_row(self,document):
        '''
        Pre-process the corpus by converting each word to lower case, removing
        special characters and apply the remove_stop_words method

        Parameters:
        ------------------------
        document: The corpus to operate on

        Returns
        -------
            Clean corpus
        '''
        document = document.lower()
        document = " ".join([re.sub('[^A-Za-z]+', ' ', word) for word in document.split()])
        document = self.remove_stop_words(document)

        return document

    def embedding_gen(self,data):
        '''
        Preprocess and get document embeddings based on the corpus
        Parameters:
        ------------------------
        data: DataFrame containing all text values
        Returns
        -------
            A generator object with a list of mean embeddings
        '''
        # models
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        for index, row in data.iterrows():
            document = self.preprocess_document_row(row['request__c_organisation_notes__c'].strip())
            embeddings = model.encode(document)
            yield [embeddings]

    def flatten(self,list_of_lists):
        '''
        Flatten out word embeddings depending on the number of tokens extracted from a corpus
        The number of tokens is usually arbitrary
        Parameters:
        ------------------------
        list_of_lists: Could be a list or multiple lists depending on the tokens
        Returns
        -------
            A single list containing the mean embeddings of a corpus
        '''

        return [item for sublist in list_of_lists for item in sublist]

    def embedding_dataframe(self,data):
        '''
        Generate embeddings dataframe with n-dimension based on the transformer model
        '''

        EMBEDDING_VECTOR_DIMENSION = 384
        emb_cols = [f'emb_{i}' for i in range(EMBEDDING_VECTOR_DIMENSION)]
        df_datasetembeddings = pd.DataFrame(self.flatten(list(self.embedding_gen(data))), columns=emb_cols)
        return df_datasetembeddings

    # def preprocess_dataframe(self,data: pd.DataFrame):
    #     '''
    #     Preprocess and get document embeddings based on the corpus
    #
    #     Parameters:
    #     ------------------------
    #
    #     data: DataFrame containing all text values
    #
    #     Returns
    #     -------
    #         A generator object with a list of mean embeddings
    #     '''
    #
    #     model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    #     # processed_embeddings = []
    #     # for index, row in self.data.iterrows():
    #     #     document = self.preprocess_document_row(row['request__c_organisation_notes__c'].strip())
    #     #     embeddings = model.encode(document)
    #     #     processed_embeddings.append(embeddings)
    #     # return processed_embeddings
    #     return [model.encode(self.preprocess_document_row(row['request__c_organisation_notes__c'].strip())) for index, row in data.iterrows()]


    # def generate_embeddings(self,data: pd.DataFrame):
    #     '''
    #     Generate embeddings dataframe with n-dimension based on the transformer model
    #     '''
    #
    #     embedding_vector_dimension = len(data.columns)
    #     emb_cols = [f'emb_{i}' for i in range(embedding_vector_dimension)]
    #     df_dataset_embeddings = pd.DataFrame(list(zip(self.preprocess_dataframe(data))), columns=emb_cols)
    #     return df_dataset_embeddings


    def merge_emb(self,data):
        '''
        Merge embeddings back to the original dataframe
        '''

        emb_frame = self.embedding_dataframe(data)
        emb_frame = emb_frame.fillna(0)
        mergedDf = pd.concat([data.reset_index(drop=True), emb_frame.reset_index(drop=True)], axis=1)
        mergedDf = mergedDf.drop(['request__c_organisation_notes__c'], axis=1)
        return mergedDf

    def cosine_similairty(self,sample):
        '''
        Getting the cosine similarity between embeddings
        '''
        top_n = 3
        embedding = self.merge_emb(self.data)
        cleaned_embeddings = embedding.drop(['request__c_keywords__c'], axis=1)
        sample_embed = self.merge_emb(sample)
        distances = cosine_similarity(sample_embed, cleaned_embeddings)
        # print(distances)
        top_n_dist = distances.argsort()[0][-top_n:]
        # print('length of df', len(df))
        keywords = [self.data.iloc[index, 1] for index in top_n_dist]
        return list(set(kw for keyword in keywords for kw in keyword.split(';')))
