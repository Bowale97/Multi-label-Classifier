from build_model import ModelBuilder
import pandas as pd
from gbq_connection import google_cloud_connection
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime as dt

if __name__ == "__main__":
    ''' We want the input data to be new records that have changed in the last 24 hours.
     And the output is in a dictionary, convert it to a data frame and upload it to a table with a datestamp
    '''
    gcloud_con = google_cloud_connection()
#     query = """Query a text columns needed"""

    data = gcloud_con.download_df('', query=query)
    # test_data = pd.read_excel(r'C:\Users\BowaleMusa\OneDrive\Downloads\Book1.xlsx')
    extractor = ModelBuilder()
    # embedding = extractor.cosine_similairty(test_data)
    # print(type(embedding))
    # print(embedding)
# Extracting the keywords from existing tags in our records
    keywords_dict = {index: extractor.cosine_similairty(row.to_frame().T) for index, row in data.iterrows()}
    # for index, row in test_data.iterrows():
    #     embedding = extractor.cosine_similairty(row)
    #     embeddings_dict[index] = embedding
    #     print("keywords:", embedding)

    print(keywords_dict)
# Converting the keywords extracted to a dataframe
#     a = keywords_dict
#     df = pd.DataFrame.from_dict(a, orient='index')
#     df = df.transpose()
#     df['timestamp'] = dt.now()



