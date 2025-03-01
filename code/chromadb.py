!pip install chromadb
import chromadb
client = chromadb.PersistentClient(r'/content/drive/MyDrive/introduction-to-embeddings-with-the-openai-api/chromadb')
collections = client.list_collections()
print(collections)

for collection in collections:
    client.delete_collection(collection)

collections = client.list_collections()
print(collections)


from chromadb.utils import embedding_functions
embedding_functions = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name = 'chromadb_collection',
    embedding_function= embedding_functions
)


client.list_collections()



llms_list = [
    {
        "name": "GPT-4",
        "description": "The fourth iteration of OpenAI's Generative Pretrained Transformer, capable of generating human-like text based on given prompts."
    },
    {
        "name": "BERT",
        "description": "Bidirectional Encoder Representations from Transformers by Google. BERT is designed to understand the context of words in a sentence for natural language understanding tasks."
    },
    {
        "name": "T5",
        "description": "Text-To-Text Transfer Transformer by Google. T5 is a transformer model that converts all NLP tasks into a text-to-text format."
    },
    {
        "name": "RoBERTa",
        "description": "Robustly Optimized BERT Pretraining Approach by Facebook AI. RoBERTa improves BERT by training it on more data with larger batch sizes and longer sequences."
    },
    {
        "name": "GPT-3",
        "description": "The third iteration of OpenAI's Generative Pretrained Transformer, known for its impressive ability to generate coherent and contextually relevant text based on input prompts."
    },
    {
        "name": "XLNet",
        "description": "Generalized autoregressive pretraining for language understanding by Google. XLNet incorporates the best of both autoregressive models and BERT, capturing bidirectional context and leveraging permutations."
    },
    {
        "name": "T5.1.1",
        "description": "An improved version of T5 by Google, with enhancements in training and performance for text-to-text tasks."
    },
    {
        "name": "ChatGPT",
        "description": "OpenAI's conversational AI model designed for dialogue applications, based on the GPT-3 framework."
    },
    {
        "name": "ALBERT",
        "description": "A Lite BERT by Google. ALBERT is a lighter version of BERT that reduces the number of parameters while maintaining performance."
    },
    {
        "name": "DistilBERT",
        "description": "A smaller, faster, cheaper version of BERT by Hugging Face. DistilBERT retains 97% of BERT's language understanding capabilities while being more efficient."
    },
    {
        "name": "CTRL",
        "description": "A Conditional Transformer Language model by Salesforce. CTRL is designed for controllable text generation by conditioning on control codes."
    }
]


ids = []
documents = []
for idx , llm in enumerate(llms_list):
  # for i, row in enumerate(llm):
    # print('row',row)
    ids.append(llm['name'])
    text = f"name: {llm['name']}, Description: {llm['description']}"
    documents.append(text)



collection.add(
    ids = ids ,
    documents  = documents
)


results = collection.query(
    query_texts= ["AI model for text summarization"],
    n_results = 2
)


# Retrieve the documents for the reference_ids
reference_ids = collection.get(ids = ['ChatGPT', 'T5.1.1'])['documents']
reference_ids



collection.update(
    ids=['GPT-4', 'BERT'],
    metadatas=[
        {'release_year': 2023, 'company': 'OpenAI'},
        {'release_year': 2018, 'company': 'Google'}
    ]
)


collection.get(ids = ['GPT-4'])



recommendations_result = collection.query(
    query_texts= reference_ids,
    n_results= 3,
where={
"release_year": {
"$eq": 2023}    })


recommendations_result
