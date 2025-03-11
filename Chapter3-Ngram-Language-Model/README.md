# n_grams

Chạy lệnh sau để train
python n_grams.py --mode train \
                  --model_type normal \
                  --n 3 \
                  --train_file train.txt \
                  --valid_file valid.txt \
                  --model_path model.pkl

# chạy lệnh sau để infer
python n_grams.py --mode inference \
                  --model_type normal \
                  --model_path model.pkl \
                  --input_text "mô hình ngôn ngữ" \
                  --max_words 10
