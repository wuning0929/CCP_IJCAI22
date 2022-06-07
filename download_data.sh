set -x

mkdir data & cd data

# NQ dataset

mkdir nq & cd nq

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz

gunzip biencoder-nq-dev.json.gz
gunzip biencoder-nq-train.json.gz

cd ..

# Mr.TyDi dataset

mkdir mrtydi & cd mrtydi

wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-arabic.tar.gz
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-bengali.tar.gz
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-english.tar.gz
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-finnish.tar.gz 
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-indonesian.tar.gz 
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-japanese.tar.gz
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-korean.tar.gz 
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-russian.tar.gz
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-swahili.tar.gz 
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-telugu.tar.gz 
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-thai.tar.gz

tar -xvzf mrtydi-v1.1-arabic.tar.gz
tar -xvzf mrtydi-v1.1-bengali.tar.gz
tar -xvzf mrtydi-v1.1-english.tar.gz
tar -xvzf mrtydi-v1.1-finnish.tar.gz
tar -xvzf mrtydi-v1.1-indonesian.tar.gz
tar -xvzf mrtydi-v1.1-japanese.tar.gz
tar -xvzf mrtydi-v1.1-korean.tar.gz
tar -xvzf mrtydi-v1.1-russian.tar.gz
tar -xvzf mrtydi-v1.1-swahili.tar.gz
tar -xvzf mrtydi-v1.1-telugu.tar.gz
tar -xvzf mrtydi-v1.1-thai.tar.gz

rm *.tar.gz

mv mrtydi-v1.1-arabic ar
mv mrtydi-v1.1-bengali bn
mv mrtydi-v1.1-english en
mv mrtydi-v1.1-finnish fi
mv mrtydi-v1.1-indonesian id
mv mrtydi-v1.1-japanese ja
mv mrtydi-v1.1-korean ko
mv mrtydi-v1.1-russian ru
mv mrtydi-v1.1-swahili sw
mv mrtydi-v1.1-telugu te
mv mrtydi-v1.1-thai th

gunzip mrtydi-v1.1-ar/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-bn/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-en/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-fi/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-id/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-ja/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-ko/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-ru/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-sw/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-te/collection/docs.jsonl.gz
gunzip mrtydi-v1.1-th/collection/docs.jsonl.gz

cd ..
