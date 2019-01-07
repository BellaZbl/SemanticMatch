#!bash/bin

python train.py \
--vocab_path=./data/vocab.txt \
--new_vocab=False \
--use_tfrecord=True \
--new_tfrecord=False \
--model_name=transformer \
--train_dataset=./data/sim_train_char_.txt \
--test_dataset=./data/sim_test_char_.txt \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--restore_model=/output/transformer_2kw.ckpt