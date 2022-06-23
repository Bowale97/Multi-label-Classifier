from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import storage
import pandas_gbq as gbq
import pandas as pd
import os

class google_cloud_connection(object):

    def __init__(self):
        self.credentials = service_account.Credentials.from_service_account_file("C:\\Users\\BowaleMusa\\PycharmProjects\\pythonProject6\\authentication.json")

        self.project_id = 'analytics-240815'
        self.bq_client = bigquery.Client(credentials=self.credentials, project=self.project_id,location='eu')
        self.storage_client = storage.Client(credentials=self.credentials,project=self.project_id)

    def upload_file(self,bucket_name, source_file_name):

        """Uploads a file to the bucket."""
        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_file_name)
        blob.upload_from_filename(source_file_name)
        print (source_file_name + ' uploaded')

    def rename_blob(self,bucket_name, blob_name, new_name):
        """Renames a blob."""
        storage_client = self.storage_client
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        new_blob = bucket.rename_blob(blob, new_name)

        print('Blob {} has been renamed to {}'.format(
            blob.name, new_blob.name))

    def upload_string(self,bucket_name,id,data):
        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(str(id) + '.txt')
        blob.upload_from_string(data)
        print(str(id) + ' uploaded')

    def download_blob(self,bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        # bucket_name = "your-bucket-name"
        # source_blob_name = "storage-object-name"
        # destination_file_name = "local/path/to/file"

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        destination_full_path = destination_file_name
        destination_subdir = os.path.dirname(destination_full_path)

        if '/user_code' in destination_subdir:
            destination_subdir = destination_subdir.replace('/user_code', '')
        if '/user_code' in destination_full_path:
            destination_subdir = destination_full_path.replace('/user_code', '')
        if not os.path.exists(destination_subdir):
            os.makedirs(destination_subdir)
        blob.download_to_filename(destination_full_path)

       # print(
       #     "Blob {} downloaded to {}.".format(
       #         source_blob_name, destination_full_path
       #     )
       # )

    def export_items_to_bigquery(self,table,dict_):
        # Instantiates a client


        # Prepares a reference to the dataset
        dataset_ref = self.bq_client.dataset(self.project_id)

        table_ref = dataset_ref.table(table)
        table = self.bq_client.get_table(table_ref)  # API call

        rows_to_insert = []
        for k,v in dict_.items():
            row = (k,v)
            rows_to_insert.append(row)


        rows_to_insert = [
            (u'Phred Phlyntstone', 32),
            (u'Wylma Phlyntstone', 29),
        ]
        errors = self.bq_client.insert_rows(table, rows_to_insert)  # API request
        assert errors == []

    def get_file(self):
        bucket_name = os.environ.get('BUCKET_NAME',
                                     app_identity.get_default_gcs_bucket_name())

        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Demo GCS Application running from Version: '
                            + os.environ['CURRENT_VERSION_ID'] + '\n')
        self.response.write('Using bucket name: ' + bucket_name + '\n\n')


    def get_last_id(self):
        try:
            query_job = self.bq_client.query("""
              SELECT
                max(script_id)
              FROM scripts.index
              """)

            results = query_job.result()  # Waits for job to complete.
            for row in results:
                print(row)
            last_id = row['f0_']
            return last_id
        except Exception as e:
            print(e)
            return 0

    def upload_df(self, df, table_name,if_exists='append'):
        try:
            df.to_gbq(table_name, project_id=self.project_id, credentials=self.credentials, location='EU',if_exists=if_exists)
        except:
            gbq.to_gbq(df, destination_table=table_name, project_id=self.project_id, credentials=self.credentials,
                       location='EU',if_exists=if_exists)
        print('Table uploaded')

    def download_df(self,table_name,filter_dict=False,query=False):

        if filter_dict != False:
            WHERE = " WHERE "

            for k,v in filter_dict.items():
                WHERE = WHERE + k + ' = ' + str(v)
        else:
            WHERE = ''

        if not query:
            query = "SELECT * FROM " + table_name + WHERE
        df = pd.read_gbq(query,project_id=self.project_id,credentials=self.credentials,location='EU')
        return df