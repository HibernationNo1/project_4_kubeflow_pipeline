python -m memory_profiler pipeline.py \
--cfg_pipeline config/pipeline.py \
--pipeline_v 0.32 \
--dashboard_pw 4958  \
--cfg_train config/train_cfg.py \
--model MaskRCNN \
--pipeline_n train
