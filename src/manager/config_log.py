
from pymongo import MongoClient
from bson.objectid import ObjectId
import pprint
import datetime


client = MongoClient('localhost', 27017)

db = client.training_pipeline
collection = db.configs
print(db.collection_names(include_system_collections=False))


def save_to_db(conf, results, hash):

	new_log = conf.__dict__
	new_log['_id'] = ObjectId()
	new_log['results'] = results
	new_log['hash'] = hash
	return collection.insert_one(new_log).inserted_id


def delete_all():
	return collection.delete_many({})

def find_all():
	return collection.find_one()
