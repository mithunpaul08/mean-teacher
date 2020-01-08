head -100 data/rte/fever/train/fever_train_delex.jsonl > temp
mv temp data/rte/fever/train/fever_train_delex.jsonl
head -100 data/rte/fever/train/fever_train_lex.jsonl > temp
mv temp data/rte/fever/train/fever_train_lex.jsonl
head -20 data/rte/fever/dev/fever_dev_delex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_delex.jsonl
head -20 data/rte/fever/dev/fever_dev_lex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_lex.jsonl
