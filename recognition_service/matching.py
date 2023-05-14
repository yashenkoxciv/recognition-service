from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema


class Matching:
    def __init__(self, milvus_uri, user, password, collection_name):
        connections.connect(
            "default",
            uri=milvus_uri,
            user=user,
            password=password,
            secure=True
        )

        self.collection_name = collection_name

        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
        fv_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1280)

        self.schema = CollectionSchema(
            fields=[id_field, fv_field],
            auto_id=True,
            description="image classification data"
        )

        self.collection = Collection(name=self.collection_name, schema=self.schema)

        self.collection.load()

        self.search_params = {"metric_type": "L2"}

    
    def get_knn(self, q_vec: list[float], topk: int = 1):
        results = self.collection.search(
            [q_vec],
            anns_field='vector',
            param=self.search_params,
            limit=topk,
            guarantee_timestamp=1
        )
        return results

