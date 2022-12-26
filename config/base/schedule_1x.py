# learning policy

# epochs단위로 설정 시 training time의 단위가 크다.
# max_iters 사용시 iteration단위로 train
runner = dict(max_epochs=12)        # EpochBasedRunner



# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
