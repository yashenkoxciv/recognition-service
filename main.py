import os
import pika
import math
import json
from recognition_service.inference import Model
from recognition_service.matching import Matching
from recognition_service.misc import load_image
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

#client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
client = MongoClient(os.environ["MONGODB_URL"])
db = client.autorec

model = Model()
matching = Matching(os.environ["MILVUS_URI"], os.environ["MILVUS_USERNAME"], os.environ["MILVUS_PASSWORD"], 'image_cats')

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=os.environ["MQ_HOST"]))

channel = connection.channel()

channel.queue_declare(queue='rpc_queue')
channel.queue_declare(queue='clustering_queue')

def recognize(image_url):
    image = load_image(image_url)
    f = model.extract_features(image)

    res = matching.get_knn(f)
    
    vector_id = None
    if len(res) > 0:
        dist = math.sqrt(res[0].distances[0])
        idx = res[0].ids[0]

        if dist < 10.413:
            vector_id = idx
    
    # TODO: find subcategory with vector_id field equals vector_id value
    # put subcategory id and category id in response

    subcategory_id = None
    category_id = None
    if vector_id is not None:
        subcategory = db["subcategory"].find_one({'vector_id': vector_id})
        if subcategory:
            subcategory_id = str(subcategory['_id'])
            category_id = str(subcategory['category_id'])
    result = {'subcategory_id': subcategory_id, 'category_id': category_id}
    return result, f

def on_request(ch, method, props, body):
    n = body.decode('utf-8')
    message = json.loads(n)

    print(" [.] recognize(%s)" % n)
    response, fv = recognize(message['image_url'])

    response_json = json.dumps(response)
    print(response_json)
    

    if response['subcategory_id'] is None:
        print(message['image_id'], 'sent to clustering')
        # send to clustering
        clustering_message = {
            'image_id': message['image_id'],
            'feature_vector': fv
        }
        clustering_message_json = json.dumps(clustering_message)
        ch.basic_publish(
            exchange='',
            routing_key='clustering_queue',
            body=clustering_message_json
        )

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str(response_json))
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()



