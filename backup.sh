COSME_DB_NAME=cosme_scan_db

BACKUP_DIR=cosme_db_backup
ES_BACKUP_DIR=cosme_scan_es

MONGO_HOST=localhost
MONGO_PORT=27017

ES_HOST=localhost
ES_PORT=9200

mkdir $BACKUP_DIR

mongodump --host $MONGO_HOST --port $MONGO_PORT --db $COSME_DB_NAME --out $BACKUP_DIR

mkdir -p $BACKUP_DIR/$ES_BACKUP_DIR

for es_index in cosme_scanmodel cosme_barcodemodel cosme_brandmodel cosme_categorymodel cosme_predictionmodel cosme_productmodel cosme_subcategorymodel
do
    elasticdump --input=http://$ES_HOST:$ES_PORT/$es_index --output=$ES_BACKUP_DIR/$es_index.json --type=data
done