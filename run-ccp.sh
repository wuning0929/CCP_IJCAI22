outdir=runs/CCP
model_cfg=data/ccp

# python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
# --max_grad_norm 2.0 --encoder_model_type hf_bert --pretrained_model_cfg ${model_cfg} \
# --seed 42 --sequence_length 256 --warmup_steps 1237 --batch_size 6 --dev_batch_size 32 \
# --train_file data/nq/biencoder-nq-train.json --dev_file data/nq/biencoder-nq-dev.json \
# --output_dir ${outdir} --learning_rate 1e-05 --num_train_epochs 20 --fp16 --pooling cls

### mrtydi
# for lang in sw bn te th id ko fi ar ja ru en; do
# 	for i in {0..7}; do
# 		CUDA_VISIBLE_DEVICES=${i} python generate_dense_embeddings.py --model_file ${outdir}/best.pt --shard_id ${i} --num_shards 8 --fp16  \
# 		--ctx_file data/mrtydi/${lang}/collection/docs.jsonl --out_file ${outdir}/emb_mrtydi/${lang}/emb  &
# 	done
# 	wait

# 	CUDA_VISIBLE_DEVICES=0 python dense_retriever.py --data mrtydi --model_file ${outdir}/best.pt --fp16 \
# 	--encoded_ctx_file ${outdir}/emb_mrtydi/${lang}/emb_\* --log_file ${outdir}/mrtydi/logger.log --n-docs 100 \
# 	--input_file data/mrtydi/${lang}/topic.test.tsv --out_file ${outdir}/mrtydi/test_${lang}.tsv 
# done

### xor-retrieve
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=${i} python generate_dense_embeddings.py --model_file ${outdir}/best.pt --shard_id ${i} \
  --ctx_file data/xorqa/en_wiki.tsv --out_file ${outdir}/emb_xorqa/emb --num_shards 8 --fp16 &
done
wait

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py --data xorqa --n-docs 100 --fp16 --model_file ${outdir}/best.pt \
--encoded_ctx_file ${outdir}/emb_xorqa/emb_\*  --ctx_file data/xorqa/en_wiki.tsv --log_file ${outdir}/xorqa/logger.log \
--input_file data/xorqa/dev.jsonl data/xorqa/test.jsonl --out_file ${outdir}/xorqa/dev.json ${outdir}/xorqa/test.json
