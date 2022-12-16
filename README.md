# CS692 course project
Code base is https://github.com/tonyzhaozh/few-shot-learning
Dataset under data/ are from https://github.com/jkkummerfeld/text2sql-data

### Data
Only public dataset can be used.


### Run experiment
```
python run_no_calibrate.py  --model="text-davinci-003" --dataset="cmput291" --num_seeds=1 --all_shots=0 --subsample_test_set=1 --api_num_log_prob=10
```


# References
Please follow cite instruction in https://github.com/jkkummerfeld/text2sql-data and https://github.com/tonyzhaozh/few-shot-learning