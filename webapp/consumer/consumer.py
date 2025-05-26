import logging
import asyncio
from json import loads
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import json
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from typing import AsyncIterator
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
import markdown
from bs4 import BeautifulSoup
from textwrap import dedent
# Asynchronous stream function for tokens
model=Ollama(id="llama3.2:1b", host="localhost:11434")
embed_model = SentenceTransformer('all-mpnet-base-v2')

async def data_stream(content):
    rag_info = ''
    es = Elasticsearch(
    "http://localhost:9200",
)
    if es.ping():
        print('Connected to ES!')
    else:
        print('Could not connect to ES!')
        exit(1)
    max_candi = es.count(index="rag")["count"]
    if max_candi > 0:
        query = {
        "field" : "SummaryVector",
        "query_vector" : embed_model.encode(content),
        "k" : 5,
        "num_candidates" : max_candi , 
    }
        res = es.knn_search(index="lang1", knn=query , source=["Content","Summary"])
        rag_info = '\n'.join(item['_source']['Content'] for item in res["hits"]["hits"])
    
    llm = Agent(model=model, 
                context = {"rag":rag_info },
                instructions=dedent("""\
                    You are VERON, IC design assistance. You can decide to answer base on the reference or not. 📰

                    Here are some reference information from documents you can reference:
                    {rag}\
                """),   
                # show_tool_calls = True,
                markdown=True, 
                )

    run_response: AsyncIterator[RunResponse] = await llm.arun(content, stream = True)
    
    try:
        async for response in run_response:
            if isinstance(response, RunResponse):
                # Extract the content from the RunResponse object
                response_content = response.content
                yield response_content
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        print("Stream success")
        
# The function that sends the stream of messages to Kafka
async def send_stream_of_messages(content, key):
    # Initialize the Kafka producer
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8')  # Serialize the key
    )
#     try:
#         conversation = {"role": "user", "content": content}
#         es.index(index="tryagno", document={"user_id": user_id, "timestamp": datetime.now(timezone.utc) , 
#         "messages": conversation, "messages_vector": embed_query(conversation["content"])})
#         print("Debugging before AI response")

# # print(" Data successfully stored in Elasticsearch!")
#     except Exception as e:
#         print(f"⚠️ Error saving to Elasticsearch: {e}")

    # Start the producer
    await producer.start()
    mess = ""
    try:
        # Get the token stream from data_stream
        async for token in data_stream(content):
            # Send each token as a message to the Kafka topic 'response'
            await producer.send_and_wait('response', value={'token': token}, key=key)
            logging.info(f"Sent token: {token}")  # Logging for debugging
            mess += token
        # conversation = {"role": "assistant", "content": mess}
        # es.index(index="tryagno", document={"user_id": user_id, "timestamp": datetime.now(timezone.utc) , 
        # "messages": conversation, "messages_vector": embed_query(conversation["content"])})

    except Exception as e:
        logging.error(f"Error sending messages: {e}")
    finally:
        # Stop the producer gracefully
        await producer.stop()


# Function to consume messages from Kafka
async def consume_messages():
    # Initialize the consumer
    consumer = AIOKafkaConsumer(
        'hello',  # Topic to consume from
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda x: loads(x.decode('utf-8')),  # Deserialize JSON values
        enable_auto_commit=False,  # Disable auto commit to manually commit offsets
        max_poll_records=100  # Maximum number of records to fetch in a single poll
    )

    # Start the consumer
    await consumer.start()

    try:
        # Consume messages
        async for message in consumer:
            print(f"{message.topic}:{message.partition}:{message.offset}: key={message.key} value={message.value}")
            # Decode the key if present
            id = message.key.decode() if message.key else None
            print("ID: ", id)
            
            
            # Call send_stream_of_messages to process tokens
            await send_stream_of_messages(message.value, id)

            # Manually commit the offset after processing the message
            # await consumer.commit()
            logging.info(f"Committed offset {message.offset + 1} for partition {message.partition}")
    except Exception as e:
        logging.error(f"Error consuming message: {e}")
    finally:
        # Stop the consumer gracefully
        await consumer.stop()

if __name__ == '__main__':

    # es = Elasticsearch("http://localhost:9200")
    # if es.ping():
    #     print('Connected to ES!')
    # else:
    #     print('Could not connect to ES!')
    #     exit(1)
    # user_id = "agno"
    # conversation = []
    # index_mapping = {
    #         "properties": {
    #             "user_id": {"type": "keyword"}, 
    #             "timestamp": {"type": "date"},  # Time
    #             "messages": {
    #                 "type": "nested",  #stack_conversation
    #                 "properties": {
    #                     "role": {"type": "keyword"},  # "user" / "assistant"
    #                     "content": {"type": "text"}   # Nội dung 
    #                 }
    #             },
    #             "messages_vector": {
    #                 "type":"dense_vector",
    #                 "dims": 768,
    #                 "index":True,
    #                 "similarity": "l2_norm"
    #             }
    #         }
    # }
    # # ------------------------------
    # index_name = 'tryagno'
    # if es.indices.exists(index=index_name):
    #     es.indices.delete(index=index_name)
    #     #   print(f"Index '{index_name}' deleted.")
    #     es.indices.create(index=index_name, mappings=index_mapping)
    
    # embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    
    
    # inf = Agent(model= model)
    # inf.print_response("Hello world")
    
    try:
        asyncio.run(consume_messages())
    except Exception as e:
        logging.error(f'Connection failed: {e}')
