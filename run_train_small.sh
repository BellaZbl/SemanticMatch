#!bash/bin
python train.py \
--vocab_path=./data/vocab.txt \
--new_vocab=False \
--model_name=siamese_cnn  \
--train_dataset=./data/sim_train_char_small.txt \
--test_dataset=./data/sim_test_char_small.txt \
--batch_size=256 \
--dropout=0.0 \
--task_name=QQP \
--model_name=mp \
--lr=0.0001 \




