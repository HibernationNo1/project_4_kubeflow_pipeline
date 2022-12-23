model = dict(
    MaskRCNN = dict(
        backbone = 'SwinTransformer',
        neck = 'FPN'
    ),
    
    SOLO_V2 = dict(
        backcone = 'ResNeXt',
    ),
    
    Mask2Former = dict(
        backbone = 'SwinTransformer'
    )
   
)