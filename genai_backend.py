
import pandas as pd
import oracledb,oci,json,base64,redis
from collections import deque
from ocifs import OCIFileSystem
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLSectionSplitter
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder

)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os,tempfile,shutil
load_dotenv()
(DBHOST,DBUNAME,DBPASS,DBSERVICE,INSTANT_CLIENT_PATH,COMPARTMENT_ID,DOC_OCR_BUCKET,DOC_OCR_BUCKET_PREFIX,
GENAI_SERVICE_ENDPOINT,DOC_INPUT_BUCKET_PREFIX,OCI_REDIS_URL,LOCAL_REDIS_URL,ENV_TYPE,REDIS_CHAT_NAME_PREFIX) = (
os.getenv("DBHOST"),os.getenv("DBUNAME"),os.getenv("DBPASS"),os.getenv("DBSERVICE"),os.getenv("INSTANT_CLIENT_PATH"),os.getenv("COMPARTMENT_ID"),
os.getenv("DOC_OCR_BUCKET"),os.getenv("DOC_OCR_BUCKET_PREFIX"),os.getenv("GENAI_SERVICE_ENDPOINT"),os.getenv("DOC_INPUT_BUCKET_PREFIX"),
os.getenv("OCI_REDIS_URL"),os.getenv("LOCAL_REDIS_URL"),os.getenv("ENV_TYPE"),os.getenv("REDIS_CHAT_NAME_PREFIX"))

REDIS_URL = OCI_REDIS_URL if ENV_TYPE=="PROD" else LOCAL_REDIS_URL

OCI_CHAT_MODELS =["cohere.command-r-16k","cohere.command-r-plus","meta.llama-3-70b-instruct"]
OCI_GEN_MODELS = ["cohere.command","cohere.command-light","meta.llama-2-70b-chat"]
ALLOWED_FILE_TYPES = ["pdf","txt","html"]
CONFIG = oci.config.from_file()
OBJECT_STORAGE_CLIENT = oci.object_storage.ObjectStorageClient(CONFIG)
AIDOC_CLIENT = oci.ai_document.AIServiceDocumentClient(CONFIG)
AIDOC_COMP_CLIENT = oci.ai_document.AIServiceDocumentClientCompositeOperations(AIDOC_CLIENT)
NAMESPACE_NAME = OBJECT_STORAGE_CLIENT.get_namespace().data
fs = OCIFileSystem(CONFIG)
BUCKET_FILE_PATH_PREFIX = f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/{DOC_INPUT_BUCKET_PREFIX}/"
BUCKET_OCR_PATH_PREFIX = f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/{DOC_OCR_BUCKET_PREFIX}/"
oracledb.init_oracle_client(INSTANT_CLIENT_PATH)

def __get_conn():
    return oracledb.connect(
        host=DBHOST,
        user=DBUNAME,
        password=DBPASS,
        service_name=DBSERVICE
    )


def __delete_object_store_folder(path):
    fs.invalidate_cache()
    if(fs.exists(path)):
        try:
            fs.rm(path,recursive=True)
        except Exception as e:        
            pass
        for  x in fs.listdir(path):
            __delete_object_store_folder(f"oci://{x['name']}")
            try:
                OBJECT_STORAGE_CLIENT.delete_object(namespace_name=NAMESPACE_NAME,bucket_name=DOC_OCR_BUCKET,object_name=f"{x['name'].split('/',1)[1]}/")
            except Exception as e:
                pass

def get_document_store_sources(doc_store):
    fs.invalidate_cache()
    path = BUCKET_FILE_PATH_PREFIX+doc_store
    l = [f"oci://{y['name']}" for x in fs.listdir(path) for y in fs.listdir(f"oci://{x['name']}")]
    return l



def truncate_document_store(doc_store):
    try:        
        conn = __get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE {doc_store}")
        conn.commit()
        conn.close()
        __delete_object_store_folder(BUCKET_FILE_PATH_PREFIX+doc_store)
        __delete_object_store_folder(BUCKET_OCR_PATH_PREFIX+doc_store)
    except Exception as e:
        conn.close()
        raise e


def register_document_store(name,description,is_search_engine_store=False):
    try:
        name=name.upper()
        conn = __get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"INSERT INTO DOCUMENT_STORE_LIST VALUES(:1,:2,:3)",[name,description,is_search_engine_store])
        conn.commit()
        conn.close()
    except Exception as e:
        conn.close()
        raise e

def edit_document_store(name,description,is_search_engine_store):
    try:
        conn = __get_conn()
        query1 = """
        UPDATE DOCUMENT_STORE_LIST
        SET
            DESCRIPTION_TEXT = :description,
            IS_SEARCH_ENGINE_STORE= :is_search_engine_store            
        WHERE DOC_STORE = :name
        """
        with conn.cursor() as cursor:
            cursor.execute(query1, {'description': description, 'is_search_engine_store':is_search_engine_store,"name":name})
        # if new_name!=old_name:
        #     query2 = f"ALTER TABLE {old_name} RENAME TO {new_name}"
        #     with conn.cursor() as cursor:
        #         cursor.execute(query2)
        #     fs.mv(BUCKET_FILE_PATH_PREFIX+old_name+"*",BUCKET_FILE_PATH_PREFIX+new_name+"*")
        conn.commit()
        conn.close()
    except Exception as e:
        conn.close()
        raise e

def delete_document_store(name):
    try:
        conn = __get_conn()
        query1 = "DELETE FROM DOCUMENT_STORE_LIST WHERE DOC_STORE = :name"
        query2 = f"DROP TABLE {name}"
        with conn.cursor() as cursor:
            cursor.execute(query1, {'name': name})
            cursor.execute(query2)
        conn.commit()
        conn.close()
        __delete_object_store_folder(BUCKET_FILE_PATH_PREFIX+name)
        __delete_object_store_folder(BUCKET_OCR_PATH_PREFIX+name)
        
    except Exception as e:
        conn.close()
        raise e

def delete_sources_from_document_store(doc_store,bucket_file_paths):
    fs.invalidate_cache()
    try:
        conn = __get_conn()
        in_str= ', '.join([f"'{x}'" for x in bucket_file_paths])
        query = f"DELETE FROM {doc_store} WHERE json_value(metadata, '$.source') in ({in_str})"
        with conn.cursor() as cursor:
            cursor.execute(query)    
        conn.commit()
        conn.close()
        fs.bulk_delete(bucket_file_paths)
    except Exception as e:
        conn.close()
        raise e



def get_document_stores():
    try:
        conn = __get_conn()
        query = "SELECT DOC_STORE, DESCRIPTION_TEXT, IS_SEARCH_ENGINE_STORE FROM DOCUMENT_STORE_LIST"
        df = pd.read_sql(query, con=conn)
        df['IS_SEARCH_ENGINE_STORE'] = df['IS_SEARCH_ENGINE_STORE'].astype(bool)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e

def get_search_document_stores():
    try:
        conn = __get_conn()
        query = "SELECT DOC_STORE, DESCRIPTION_TEXT, IS_SEARCH_ENGINE_STORE FROM DOCUMENT_STORE_LIST WHERE IS_SEARCH_ENGINE_STORE=TRUE"
        df = pd.read_sql(query, con=conn)
        df['IS_SEARCH_ENGINE_STORE'] = df['IS_SEARCH_ENGINE_STORE'].astype(bool)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e

def get_chat_document_stores():
    try:
        conn = __get_conn()
        query = "SELECT DOC_STORE, DESCRIPTION_TEXT, IS_SEARCH_ENGINE_STORE FROM DOCUMENT_STORE_LIST WHERE IS_SEARCH_ENGINE_STORE=FALSE"
        df = pd.read_sql(query, con=conn)
        df['IS_SEARCH_ENGINE_STORE'] = df['IS_SEARCH_ENGINE_STORE'].astype(bool)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e


def get_document_store(doc_store):
    try:
        conn = oracledb.connect(
            host=DBHOST,
            user=DBUNAME,
            password=DBPASS,
            service_name=DBSERVICE
        )
        query = f"SELECT CAST(TEXT AS VARCHAR2(4000)) TEXT,CAST(METADATA AS VARCHAR2(4000)) METADATA FROM {doc_store}"
        df = pd.read_sql(query, con=conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame([])


def __get_oracle_db_vectorstore(conn,doc_store):
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint=GENAI_SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID
    )
    vectorstore = OracleVS(
        client=conn,
        embedding_function=embeddings,
        table_name=doc_store,
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )
    return vectorstore

def __add_documents(doc_store,docs):
    try:
        conn = __get_conn()
        vectorstore = __get_oracle_db_vectorstore(conn,doc_store)
        ans = vectorstore.add_documents(docs)
        conn.close()
        return ans
    except Exception as e:
        conn.close()
        raise e

def __get_oci_genai_chat_model(model_id="cohere.command-r-16k",model_kwargs={"temperature": 0, "max_tokens": 4000}):
    return ChatOCIGenAI(
        model_id=model_id, #cohere.command-r-plus,
        service_endpoint=GENAI_SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs=model_kwargs
    )

def __get_oci_genai_gen_model(model_id="cohere.command",model_kwargs={"temperature": 0, "max_tokens": 4000}):
    return OCIGenAI(
        model_id=model_id,
        service_endpoint=GENAI_SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs=model_kwargs
    )

def push_documents_to_chat_store(doc_store,files,chunk_size=500,chunk_overlap=20):
    fs.invalidate_cache()
    docs = deque()
    temp_dir = tempfile.mkdtemp()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    for fileobj in files:
        ext = fileobj.name.split(".")[-1]
        oci_path = BUCKET_FILE_PATH_PREFIX+f"{doc_store}/{ext}/{fileobj.name}"
        with fs.open(oci_path,"wb") as f:
            f.write(fileobj.getvalue())
        if ext=="txt":                       
            data = text_splitter.split_documents([Document(page_content=fileobj.getvalue().decode(),metadata={"source":oci_path})])
            docs.extend(data)
        elif ext=="html":
            curr_temp_path=os.path.join(temp_dir, fileobj.name)           
            with open(curr_temp_path,"wb") as f:
                f.write(fileobj.getvalue())
            loader = BSHTMLLoader(curr_temp_path)
            data = loader.load()
            # headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"),("h3", "Header 3"),("h4", "Header 4"),("h5", "Header 5"),("h6", "Header 6")]
            # html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
            # data = html_splitter.split_documents(data)            
            data = text_splitter.split_documents(data)
            for x in data:
                x.metadata["source"]=oci_path
            docs.extend(data)
        elif ext=="pdf":
            curr_temp_path=os.path.join(temp_dir, fileobj.name)           
            with open(curr_temp_path,"wb") as f:
                f.write(fileobj.getvalue())
            loader = PyPDFLoader(curr_temp_path)
            data = loader.load_and_split(text_splitter)
            for x in data:
                x.metadata["source"]=oci_path            
            docs.extend(data)       
        else:
            pass
    shutil.rmtree(temp_dir)
    __add_documents(doc_store,docs)
    return docs

def intiate_search_store_ocr_job(doc_store,files,job_name):
    fs.invalidate_cache()
    object_list = deque()
    for fileobj in files:
        ext = fileobj.name.split(".")[-1]
        oci_path = BUCKET_FILE_PATH_PREFIX+f"{doc_store}/{ext}/{fileobj.name}"
        with fs.open(oci_path,"wb") as f:
            f.write(fileobj.getvalue())
        object_list.append(DOC_INPUT_BUCKET_PREFIX+f"/{doc_store}/{ext}/{fileobj.name}")

    text_extraction_feature = oci.ai_document.models.DocumentTextExtractionFeature()
    output_location = oci.ai_document.models.OutputLocation()
    output_location.namespace_name = NAMESPACE_NAME
    output_location.bucket_name = DOC_OCR_BUCKET
    output_location.prefix =  DOC_OCR_BUCKET_PREFIX+f"/{doc_store}"
    object_locations=[oci.ai_document.models.ObjectLocation(bucket_name=DOC_OCR_BUCKET,namespace_name=NAMESPACE_NAME,object_name=o) for o in object_list]
    source_type=oci.ai_document.models.ObjectStorageLocations.SOURCE_TYPE_OBJECT_STORAGE_LOCATIONS
    create_processor_job_details_text_extraction = oci.ai_document.models.CreateProcessorJobDetails(
        compartment_id=COMPARTMENT_ID,
        display_name=job_name,
        input_location=oci.ai_document.models.ObjectStorageLocations(source_type=source_type,object_locations=object_locations),
        output_location=output_location,
        processor_config=oci.ai_document.models.GeneralProcessorConfig(language="ENG",features=[text_extraction_feature])
    )
    
    proc_job = AIDOC_CLIENT.create_processor_job(
        create_processor_job_details=create_processor_job_details_text_extraction
    )

    print(f"({proc_job.data.display_name})",proc_job.data.id)
    return proc_job.data

def list_search_store_ocr_jobs(doc_store):
    fs.invalidate_cache()
    ans = deque()
    oci_dir = BUCKET_OCR_PATH_PREFIX+f"{doc_store}"
    job_ocids = [x.split("/")[-1] for x in fs.ls(oci_dir)]
    for job_ocid in job_ocids:
        try:
            res=AIDOC_CLIENT.get_processor_job(job_ocid)
            proc_files = oci.pagination.list_call_get_all_results(OBJECT_STORAGE_CLIENT.list_objects,namespace_name=NAMESPACE_NAME, bucket_name=DOC_OCR_BUCKET,prefix=DOC_OCR_BUCKET_PREFIX+f"/{doc_store}/{job_ocid}").data.objects
            proc_files = [f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/"+o.name for o in proc_files if o.name.endswith(".pdf.json")]
            obj = {
                "JOB_OCID":job_ocid,
                "JOB_NAME":res.data.display_name,
                "JOB_STATE":res.data.lifecycle_state,
                "FILES_SUBMITTED":[f"oci://{o.bucket_name}@{o.namespace_name}/{o.object_name}" for  o in res.data.input_location.object_locations],
                "FILES_PROCESSED":proc_files,
                "PARTIAL_SUCCESS": None if res.data.lifecycle_state in ("ACCEPTED", "CANCELED", "IN_PROGRESS", "CANCELING","SUCCEEDED") else  (True if res.data.lifecycle_details=="PARTIALLY_SUCCEEDED" else False)
            }
            ans.append(obj)
        except Exception as e:
            raise e
    return pd.DataFrame(ans,columns=["JOB_OCID","JOB_NAME","JOB_STATE","FILES_SUBMITTED","FILES_PROCESSED","PARTIAL_SUCCESS"])
    
def list_search_store_ocr_json(doc_store):
    fs.invalidate_cache()
    json_files = oci.pagination.list_call_get_all_results(OBJECT_STORAGE_CLIENT.list_objects,namespace_name=NAMESPACE_NAME, bucket_name=DOC_OCR_BUCKET,prefix=DOC_OCR_BUCKET_PREFIX+f"/{doc_store}").data.objects
    json_files = [f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/"+o.name for o in json_files if o.name.endswith(".pdf.json")]
    df = pd.DataFrame({"json_file":json_files})
    df["content"] = df["json_file"].apply(lambda x :json.loads(fs.cat(x)))
    return df


def process_search_store_json(doc_store,json_df):
    fs.invalidate_cache()
    INCH_TO_PIXEL = 72
    n = json_df.shape[0]
    for i in range(n):
        json_obj,source = json_df.iloc[i][["content","json_file"]].tolist()
        source = f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/{source.split('/',8)[-1][:-5]}"
        docs = deque()
        for page in json_obj["pages"]: 
            page_num = page["pageNumber"]
            page_width, page_height = page["dimensions"]["width"]*INCH_TO_PIXEL,page["dimensions"]["height"]*INCH_TO_PIXEL
            for line in page["lines"]:
                if not line["text"].strip():
                    continue
                top_left,_,bottom_right,_=line["boundingPolygon"]["normalizedVertices"]
                x,y,width,height = top_left["x"]*page_width, top_left["y"]*page_height,(bottom_right["x"]-top_left["x"])*page_width,(bottom_right["y"]-top_left["y"])*page_height
                docs.append(Document(page_content=line["text"],metadata={"page":page_num,"x":x,"y":y,"width":width,"height":height,"source":source}))                
        __add_documents(doc_store,docs)
    fs.bulk_delete(json_df["json_file"].tolist())
    


def get_rag_response(query,doc_store,model_id="cohere.command-r-16k",context_limit=30):
    try:
        conn = __get_conn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DESCRIPTION_TEXT FROM DOCUMENT_STORE_LIST WHERE DOC_STORE = :name",{"name":doc_store})
            description = cursor.fetchall()[0][0]
        vectorstore = __get_oracle_db_vectorstore(conn,doc_store)
        retriever = vectorstore.as_retriever(search_kwargs={"k":context_limit,"filter":None})
        model = __get_oci_genai_chat_model(model_id=model_id) if model_id in OCI_CHAT_MODELS else __get_oci_genai_gen_model(model_id=model_id)
        rag_prompt_template = (f"Conider a document store with the follwing description:\n{description}\n\n"
        "Context Documents from this store:\n{context}\n\nNow using it answer the follwing Question: {question}")
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        rag = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )
        res = rag.invoke(query)
        conn.close()
        return res
    except Exception as e:
        conn.close()
        raise e

def search_docs(query,doc_store,n_results=30):
    try:
        conn = __get_conn()
        vectorstore = __get_oracle_db_vectorstore(conn,doc_store)
        retriever=vectorstore.as_retriever(search_kwargs={"k":n_results,"filter":None})
        return retriever.invoke(query)
    except Exception as e:
        conn.close()
        raise e

def __get_redis_connection():
    return redis.from_url(REDIS_URL)

def get_all_chat_names():        
    r = __get_redis_connection()
    n = len(REDIS_CHAT_NAME_PREFIX)    
    pattern = REDIS_CHAT_NAME_PREFIX+"*"
    cursor = '0'
    keys_with_prefix = deque()
    while cursor != 0:
        cursor, keys = r.scan(cursor=cursor, match=pattern)
        keys_with_prefix.extend(keys)
    ans = [x.decode()[n:] for x in keys_with_prefix]
    # ans = [x.decode()[n:] for x in r.keys(REDIS_CHAT_NAME_PREFIX+"*")]
    r.close()
    return ans

def create_chat(chat_name):
    r = __get_redis_connection()
    r.lpush(f"{REDIS_CHAT_NAME_PREFIX}{chat_name}", '')
    r.ltrim(f"{REDIS_CHAT_NAME_PREFIX}{chat_name}", 1, 0)

def rename_chat(old_chat_name,new_chat_name):
    r = __get_redis_connection()
    r.rename(REDIS_CHAT_NAME_PREFIX+old_chat_name,REDIS_CHAT_NAME_PREFIX+new_chat_name)
    r.close()

def get_chat_messages(chat_name):
    chat_history = RedisChatMessageHistory(session_id=chat_name, url=REDIS_URL,key_prefix=REDIS_CHAT_NAME_PREFIX)
    messages = chat_history.messages
    chat_history.redis_client.close()
    return messages
    

def chat(chat_name,prompt,doc_store,model_id="cohere.command-r-16k",chat_history_limit=4,context_retrieval_limit=10):
    try:
        conn = __get_conn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DESCRIPTION_TEXT FROM DOCUMENT_STORE_LIST WHERE DOC_STORE = :name",{"name":doc_store})
            description = cursor.fetchall()[0][0]
        system_template_str = ("You are a helpfull chat bot who will respond to the user prompts."
        "You can make use of the documents provided as context from a docoument store with the following description:\n"+description)

        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(template=system_template_str)
        )

        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"], template="{question}"
            )
        )

        system_prompt_context = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"], template="Some Context Documents:\n{context}"
            )
        )

        messages1 = [system_prompt,
                    MessagesPlaceholder(variable_name="history"),
                    human_prompt]
        messages2 = [system_prompt,
                    MessagesPlaceholder(variable_name="history"),                
                    system_prompt_context,
                    human_prompt]
        prompt_template_rag = ChatPromptTemplate.from_messages(messages1)
        prompt_template_chat = ChatPromptTemplate.from_messages(messages2)
        # output_parser = StrOutputParser()
        
        chat_history = RedisChatMessageHistory(session_id=chat_name, url=REDIS_URL,key_prefix=REDIS_CHAT_NAME_PREFIX) 
        try:
            history_messages = chat_history.messages[-chat_history_limit:]
        except:
            history_messages = chat_history.messages
        vectorstore = __get_oracle_db_vectorstore(conn,doc_store)
        retriever=vectorstore.as_retriever(search_kwargs={"k":context_retrieval_limit,"filter":None})  
        context = retriever.invoke(prompt_template_rag.invoke({"question":prompt,"history":history_messages}).to_string())        
        chat_model = __get_oci_genai_chat_model(model_id=model_id)
        chain = ( prompt_template_chat | chat_model | StrOutputParser())
        ai_message = chain.invoke({"context":context,"question":prompt,"history":history_messages})
        chat_history.add_user_message(prompt)
        chat_history.add_ai_message(ai_message)
        chat_history.redis_client.close()           
        conn.close()
        return ai_message
    except Exception as e:
        conn.close()
        raise e