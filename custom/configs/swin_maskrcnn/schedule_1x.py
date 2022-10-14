# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(   # StepLrUpdaterHook
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# epochs단위로 설정 시 training time의 단위가 크다.
# max_iters 사용시 iteration단위로 train
runner = dict(max_epochs=12)        # EpochBasedRunner


