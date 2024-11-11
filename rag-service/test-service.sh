#!/bin/bash

curl -X POST \
  http://pop-os:8000/process_document \
  -H 'Content-Type: application/json' \
  -d '{
    "file_path": "https://www.congress.gov/118/bills/hr8785/BILLS-118hr8785ih.pdf"
}'



curl -X POST \
  http://pop-os:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the main points of this H.R. 8785?"
}'

#https://rag-service.redgrass-7c02e7f4.eastus.azurecontainerapps.io
# curl -X POST \
#   https://rag-service.redgrass-7c02e7f4.eastus.azurecontainerapps.io:8000/process_document \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "file_path": "https://www.congress.gov/118/bills/hr8785/BILLS-118hr8785ih.pdf"
# }'



# curl -X POST \
#   https://rag-service.redgrass-7c02e7f4.eastus.azurecontainerapps.io:8000/query \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "query": "What are the main points of this H.R. 8785?"
# }'