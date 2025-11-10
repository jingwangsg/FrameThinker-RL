python scripts/convert_video_holmes_to_rl_parquet_v2.py /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test.json --num-frames 8 --video-dir /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/videos/ -o /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test_v2.parquet

python scripts/convert_video_holmes_to_rl_parquet_v2.py /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train.json --num-frames 8 --video-dir /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/videos/ -o /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train_v2.parquet


python scripts/convert_to_rl_parquet.py /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test.json --num-frames 8 --media-dir /mnt/amlfs-03/shared/datasets/s3:/video_reason/ -o /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test.parquet

python scripts/convert_to_rl_parquet.py /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train.json --num-frames 8 --media-dir /mnt/amlfs-03/shared/datasets/s3:/video_reason/ -o /mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train.parquet