mkdir -p data/rte/fever/train/
mkdir -p data/rte/fever/dev/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O data/rte/fever/train/fever_train_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl -O data/rte/fever/train/fever_train_delex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl -O data/rte/fever/dev/fever_dev_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl -O data/rte/fever/dev/fever_dev_delex.jsonl

