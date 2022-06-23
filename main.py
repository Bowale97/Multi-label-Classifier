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
    query = """SELECT  organisation_notes__c as request__c_organisation_notes__c
            FROM `analytics-240815.salesforce.Request__c` 
            WHERE _sdc_batched_at > TIMESTAMP_SUB(current_timestamp(), INTERVAL 180 HOUR) AND type__c = 'Main Grant' AND disposition__c   = 'Approved'
            LIMIT 1000"""

    data = gcloud_con.download_df('', query=query)
    # test_data = pd.read_excel(r'C:\Users\BowaleMusa\OneDrive - Esmée Fairbairn Foundation\Downloads\Book1.xlsx')
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
#     gcloud_con.upload_df(df,'salesforce_output.ml_keywords',if_exists= 'append')













# df = pd.read_excel(r'C:\Users\BowaleMusa\OneDrive - Esmée Fairbairn Foundation\Downloads\Unsupervised.xlsx')

    # test_data = 'The 4Front Project is a youth organisation based on the Grahame Park Estate in Barnet, supporting young people who have experienced trauma, violence and racial injustice. Established by Temi Mwale when she was 16 following the stabbing of a friend, it supports young people who have been impacted by violence, empowering them to create change, in their own lives, communities and society. 4Front&#39;s model combines advocacy, therapeutic and leadership services and has been designed to empower young people with complex needs. In 2020 it opened the Jaheim Justice Centre, named after one of its young people who died, modelled on the Youth Justice Coalition’s Centre in Los Angeles. Like its inspiration, the Centre is open for one-to-one and group support sessions, provides advocacy, legal and Appropriate Adult services for those in contact with police and youth justice services' \
    #          ' and offers a programme of artistic, sporting and leadership training. 4Front works with 106 young people a year, its ‘members’ who agree' \
    #          ' to its principles and values. Over 90% of are from Black and racialised communities, all have been affected by a friend’s death, 80% have ' \
    #          'been victims of violence, 62% have been stopped and searched, 53% arrested and 54% excluded from school. 4Front offers members a space to discuss ' \
    #          'their experiences as well as leadership training in how to advocate for themselves and their peers. It has created a platform for members to give' \
    #          ' evidence to the Home Affairs Select Committee and the Youth Violence Commission, meetings with the Minister for Policing and Crime and the London ' \
    #          'Deputy Mayor as well as sharing their experiences on Channel 4 News and BBC Radio London.'
    # test_data_embed = merge_emb(test_data)
    # print(test_data_embed)
    # #Compute dot score between query and all document embeddings
    # scores = util.dot_score(torch.from_numpy(test_data_embed.values), torch.from_numpy(embedding.values))[0].cpu().tolist()
    #
    # #Combine docs &scores
    # doc_scores_pairs = list(zip(df,scores))
    #
    # #Sort by decreasing score
    # doc_scores_pairs = sorted(doc_scores_pairs, key=lambda x: x[1], reverse=True)
    #
    # #Output passages & scores
    # for doc,score in doc_scores_pairs:
    #     print(score,doc)

# #embedding.to_excel(r'C:\Users\BowaleMusa\OneDrive - Esmée Fairbairn Foundation\Downloads\Emdedding.xlsx')
#
# def get_document_mean_embedding(document):
#     return np.mean([merge_emb(word) for word in document.split()], axis=0)
#
#
# # Calculates the mean embedding vector of all the words in a document
#
# def cosine_similarity(vect_1, vect_2):
#     cosine_similarity_value = 1 - scipy.spatial.distance.cosine(vect_1, vect_2)
#     return cosine_similarity_value
#
#
# def get_similarity_rankings(document):
#     rankings = []
#     #     get mean document embedding of the cleaned event description i.e document arg to the function
#     document_embedding = get_document_mean_embedding(document)
#
#     # calculate cosine similarity with each text embedding
#     for embed in df:
#         cosine_similarity_value = cosine_similarity(df[embed], document_embedding)
#         rankings.append([embed, cosine_similarity_value])
#
#     rankings = sorted(rankings, key=lambda row: row[1], reverse=True)  # sort by cosine similarity in descending order
#     # return an array of [suggested_keywords, cosine_similarity] in descending order of similarity
#     return rankings
#
#
# def get_top_k_candidate_texts(limit, rankings):
#     return rankings[:limit]
#
#
# # returns the first k entries rankings list
#
# def get_actual_text_rank(actual_text, rankings):
#     #     global rankings
#     np_rankings = np.array(rankings)
#     np_rankings = np_rankings[:, 0]
#     return list(np_rankings).index(actual_text) + 1
#
#
# # returns the rank of the actual_text
# # Note that the rank is 1 more than the index at which is present in the rankings list
#
# def get_similarity_by_rank(rank, rankings):
#     return rankings[rank - 1]
#
# test_data = pd.read_excel(r'C:\Users\BowaleMusa\OneDrive - Esmée Fairbairn Foundation\Downloads\Book1.xlsx')
# test_data_embed = merge_emb(test_data)
#
# print(cosine_similarity(test_data_embed,embedding))
# # return text and the similarity by rank


# #
# #
# #Compute dot score between query and all document embeddings
# scores = util.dot_score(test_data_embed,embedding)[0].cpu().tolist()
#
# #Combine docs &scores
# doc_scores_pairs = list(zip(df,scores))
#
# #Sort by decreasing score
# doc_scores_pairs = sorted(doc_scores_pairs, key=lambda x: x[1], reverse=True)
#
# #Output passages & scores
# for doc,score in doc_scores_pairs:
#     print(score,doc)

