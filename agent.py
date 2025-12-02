import json


with open ("bank_statements.json","r") as f:
            statement = json.load(f)

print(statement)