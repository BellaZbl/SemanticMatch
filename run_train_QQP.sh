#!bash/bin
python train.py \
--vocab_path=./data/qqp_vocab.txt \
--new_vocab=True \
--model_name=siamese_cnn  \
--train_dataset=./QQP/qqa_qa_train.txt \
--test_dataset=./QQP/qqa_qa_test.txt \
--batch_size=256 \
--dropout=0.0 \
--task_name=QQP \
--model_name=mp \
--lr=0.0001 \




