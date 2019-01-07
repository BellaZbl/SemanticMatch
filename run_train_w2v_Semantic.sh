#!bash/bin
python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=True \
--use_tfrecord=True \
--new_tfrecord=False \
--model_name=transformer \
--train_dataset=./data/sim_train_word_small.txt \
--test_dataset=./data/sim_test_word_small.txt \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.0001 \


# 1000002_2000002_pairs train
change char to word
python Word2Vec/buildEmbedding.py \
--train_dataset=/data/1000002_2000002_pairs.train \
--test_dataset=/data/1000002_2000002_pairs.test \
--write_train_dataset=/data/1000002_2000002_pairs_word.train \
--write_test_dataset=/data/1000002_2000002_pairs_word.test \


python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=True \
--use_tfrecord=True \
--new_tfrecord=True \
--model_name=transformer \
--train_dataset=./data/1000002_2000002_pairs_word.train \
--test_dataset=./data/1000002_2000002_pairs_word.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.0001 \
--do_train=True \
--do_predict=False \
--cosine=True \

--restore_model=/output/transformer_Intent_Semantic_2kw.ckpt \

#infer

python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=False \
--use_tfrecord=False \
--new_tfrecord=False \
--model_name=transformer \
--train_dataset=./data/1000002_2000002_pairs.train \
--test_dataset=./data/1000002_2000002_pairs.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.0001 \
--do_train=False \
--do_predict=True \

# infer pairs
python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=False \
--use_tfrecord=False \
--new_tfrecord=False \
--model_name=transformer \
--train_dataset=./data/1000002_2000002_pairs.train \
--test_dataset=./data/1000002_2000002_pairs.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.0001 \
--do_train=False \
--do_predict=False \
--do_pairs_predict=True \
--cosine=True

# train siamses_cnn
python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=True \
--use_tfrecord=True \
--new_tfrecord=True \
--model_name=siamese_cnn \
--train_dataset=./data/1000002_2000002_pairs_word.train \
--test_dataset=./data/1000002_2000002_pairs_word.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.01 \
--do_train=True \
--do_predict=False \
--cosine=False


# train abcnn
python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=True \
--use_tfrecord=True \
--new_tfrecord=True \
--model_name=abcnn \
--train_dataset=./data/1000002_2000002_pairs_word.train \
--test_dataset=./data/1000002_2000002_pairs_word.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--model_type=ABCNN3 \
--lr=0.001 \
--do_train=True \
--do_predict=False \


# train DSSM
python train.py \
--vocab_path=./data/vocab_word.txt \
--new_vocab=True \
--use_tfrecord=True \
--new_tfrecord=True \
--model_name=dssm \
--train_dataset=./data/1000002_2000002_pairs_word.train \
--test_dataset=./data/1000002_2000002_pairs_word.test \
--use_pre_train_emb=True \
--batch_size=256 \
--dropout=0.0 \
--task_name=Intent_Semantic \
--lr=0.01 \
--do_train=True \
--do_predict=False \

