# python scripts/convert_video_holmes_to_rl_parquet_v2.py data/video_reason/Video-Holmes/test.json --num-frames 8 --video-dir data/video_reason/Video-Holmes/videos/ -o data/video_reason/Video-Holmes/test_v2.parquet

# python scripts/convert_video_holmes_to_rl_parquet_v2.py data/video_reason/Video-Holmes/train.json --num-frames 8 --video-dir data/video_reason/Video-Holmes/videos/ -o data/video_reason/Video-Holmes/train_v2.parquet


python scripts/convert_to_rl_parquet.py data/video_reason/Video-Holmes/test.json --num-frames 8 --num-proc 16 --media-dir data/video_reason/Video-Holmes/videos/ -o data/video_reason/Video-Holmes/test.parquet

python scripts/convert_to_rl_parquet.py data/video_reason/Video-Holmes/train.json --num-frames 8 --num-proc 16 --media-dir data/video_reason/Video-Holmes/videos/ -o data/video_reason/Video-Holmes/train.parquet