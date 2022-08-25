### Download data

```
bash download_data.sh
```


### Download CCP pre-trained model from google drive

https://drive.google.com/drive/folders/1jVyTfXx4qFusggaZlSxYGe96WrRUKqxU?usp=sharing


```
data
|- ccp
|  |- config.json
|  |- pytorch_model.bin
|  |- sentencepece.bpe.model
|  |- tokenizer.json
|- nq
|  |- biencoder-nq-dev.json
|  |- biencoder-nq-train.json
|- xorqa
|  |- dev.jsonl
|  |- en_wiki.tsv
|  |- test.jsonl
|  |- train.jsonl
|- mrtydi
|  |- ar
|  |   |- collection
|  |   |       |- docs.jsonl
|  |   |- pid2passage.tsv
|  |   |- qrels.dev.txt
|  |   |- qrels.test.txt
|  |   |- qrels.train.txt
|  |   |- qrels.txt
|  |   |- topic.dev.tsv
|  |   |- topic.test.tsv
|  |   |- topic.train.tsv
|  |   |- topic.tsv
|  |- bn
...
```

### Install python package

```
pip install -r requirements.txt
```

### Training and retrieval

```
bash run-ccp.sh
```

### Evaluation

```
python eval/mrtydi.py --pred_file runs/CCP/mrtydi --set test
python eval/xorqa.py --pred_file runs/CCP/xorqa/dev.json
```
