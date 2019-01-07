#!bash/bin
python Word2Vec/bulidEmbedding.py \
--origin_vocab_path=/Word2Vec/vocab_chinese.txt \
--train_dataset=/data/sim_train_char_small.txt \
--test_dataset=/data/sim_test_char_small.txt \
--write_train_dataset=/data/sim_train_word_small.txt \
--write_test_dataset=/data/sim_test_word_small.txt \
