#!/usr/bin/env sh

epochs=3
data_num=100

python main.py --clear --log_dir test --log_name b032 --batch_size 032 --epochs ${epochs} --data_num ${data_num}
python main.py --clear --log_dir test --log_name b016 --batch_size 016 --epochs ${epochs} --data_num ${data_num}

cat test/summary.csv

epochs=5
python utils/yaml_overwrite.py --log_dir test --key epochs --val ${epochs} --type int

python main.py --log_dir test --log_name b032 --batch_size 032 --epochs ${epochs} --data_num ${data_num}
python main.py --log_dir test --log_name b016 --batch_size 016 --config test/b016/config.yaml --clear

cat test/summary.csv
