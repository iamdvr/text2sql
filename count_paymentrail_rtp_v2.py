import json
import boto3

import sqlalchemy
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

from langchain.docstore.document import Document
from langchain import PromptTemplate,SagemakerEndpoint,SQLDatabase, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
#from langchain.chains import SQLDatabaseSequentialChain
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain_experimental.sql import SQLDatabaseChain


from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.chains.api import open_meteo_docs

from typing import Dict
import time

import boto3
import os

import inspect
inspect.getmodule(ChatAnthropic)

#provide user input
glue_databucket_name = 'genai-payment-processed' #Create this bucket in S3
glue_db_name='genai-payment100'
glue_role=  'AWSGlueServiceRole-gluepayment100'
glue_crawler_name=glue_db_name+'-crawler100'


region = "us-east-1" 
#os.environ['AWS_REGION']
print(region)

ANTHROPIC_API_KEY = "REPLACE_ME_WITH_ACTUAL_KEY"
#define large language model here. Make sure to set api keys for the variable ANTHROPIC_API_KEY
llm = ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, max_tokens_to_sample = 512)



#S3
# connect to s3 using athena
## athena variables
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_db_name #from user defined params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)

gdc = [schemaathena] 




#Generate Dynamic prompts to populate the Glue Data Catalog
#harvest aws crawler metadata

def parse_catalog():
    #Connect to Glue catalog
    #get metadata of redshift serverless tables
    columns_str=''

    #define glue cient
    glue_client = boto3.client('glue')

    for db in gdc:
        response = glue_client.get_tables(DatabaseName =db)
        for tables in response['TableList']:
            #classification in the response for s3 and other databases is different. Set classification based on the response location
            if tables['StorageDescriptor']['Location'].startswith('s3'):  classification='s3'
            else:  classification = tables['Parameters']['classification']
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str=columns_str+f'\n{classification}|{dbname}|{tblname}|{colname}'
    #API
    ## Append the metadata of the API to the unified glue data catalog
    columns_str=columns_str+'\n'+('api|meteo|weather|weather')
    return columns_str

glue_catalog = parse_catalog()

#display a few lines from the catalog
print('\n'.join(glue_catalog.splitlines()[-20:]) )

#Function 1 'Infer Channel'
#define a function that infers the channel/database/table and sets the database for querying
def identify_channel(query):
    #Prompt 1 'Infer Channel'
    ##set prompt template. It instructs the llm on how to evaluate and respond to the llm. It is referred to as dynamic since glue data catalog is first getting generated and appended to the prompt.
    prompt_template = """
     From the table below, find the database (in column database) which will contain the data (in corresponding column_names) to answer the question
     {query} \n
         If someone asks for the payment, they really mean the s3|genai-payment100|payment_dataset table.
     """+glue_catalog +"""
     Give your answer as database ==
     Also,give your answer as database.table ==
     """
    print("prompt_template ", prompt_template)
    ##define prompt 1
    PROMPT_channel = PromptTemplate( template=prompt_template, input_variables=["query"]  )
    print("PROMPT_channel ", PROMPT_channel)
    # define llm chain
    llm_chain = LLMChain(prompt=PROMPT_channel, llm=llm)
    #run the query and save to generated texts
    generated_texts = llm_chain.run(query)
    print("generated_texts ", generated_texts)

    #set the channel from where the query can be answered
    if 's3' in generated_texts:
            channel='db'
            db=dbathena
            print("SET database to athena")
    elif 'api' in generated_texts:
            channel='api'
            print("SET database to weather api")
    else: raise Exception("User question cannot be answered by any of the channels mentioned in the catalog")
    print("Step complete. Channel is: ", channel)

    return channel, db

#Function 2 'Run Query'
#define a function that infers the channel/database/table and sets the database for querying
def run_query(query):

    channel, db = identify_channel(query) #call the identify channel function first

    ##Prompt 2 'Run Query'
    #after determining the data channel, run the Langchain SQL Database chain to convert 'text to sql' and run the query against the source data channel.
    #provide rules for running the SQL queries in default template--> table info.

    _DEFAULT_TEMPLATE = """You are a SQL expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    Do not append 'Here is the SQL query' to SQLQuery.

    Display SQLResult after the query is run in plain english that users can understand.

    Provide answer in simple english statement.

    Only use the following tables:

    {table_info}

    Question: {input}"""

    print("_DEFAULT_TEMPLATE ", _DEFAULT_TEMPLATE)
    PROMPT_sql = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )


    if channel=='db':
        db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT_sql, verbose=False, return_intermediate_steps=False)
        response=db_chain.run(query)
    elif channel=='api':
        chain_api = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
        response=chain_api.run(query)
    else: raise Exception("Unlisted channel. Check your unified catalog")
    return response

# Enter the query
## Few queries to try out -
#athena - Healthcare - Covid dataset
# query = """How many covid hospitalizations were reported in NY in June of 2021?"""
query = """How many payments with payment rail as rtp?"""

#api - product - weather
# query = """What is the weather like right now in New York City in degrees Farenheit?"""

#Response from Langchain
response =  run_query(query)
print("----------------------------------------------------------------------")
print(f'SQL and response from user query {query}  \n  {response}')

