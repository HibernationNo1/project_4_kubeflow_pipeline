python  pipeline.py \
--cfg_pipeline config/pipeline.py \
--pipeline_v 0.1 \
--dashboard_pw 4958  \
--cfg_train config/train_cfg.py \
--model MaskRCNN \
--cfg_record config/record_cfg.py \
--cfg_eval config/evaluate_cfg.py \
--cfg_test config/test_cfg.py \
--model_path for_test/models/model_200.pth \
--pipeline_n train



