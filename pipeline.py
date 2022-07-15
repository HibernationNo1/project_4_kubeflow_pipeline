import kfp
from kfp import dsl
from kfp import onprem

import os

# image, dataset.json 을 save
def lebelme_op(pvc_name, volume_name, volume_mount_path, ratio_val):
    return dsl.ContainerOp(
        name='lebelme',
        image='{tag}/{image_name}:{version}', # TODO
        arguments=['--ann_path', volume_mount_path,
                   '--ratio-val', ratio_val],
    	).apply(
        onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def train_op(dataset_path):
    return dsl.ContainerOp(
        name='train',
        image='{tag}/{image_name}:{version}', 
        arguments=['--dataset_path', dataset_path],
        file_outputs = {'train_dataset' : '/train_dataset'}
    	)

@dsl.pipeline(
    name='ITC Pipeline',
    description=''
)    
def ITC_pipeline():

    ann_pvc_name = "annotations_path"		
    ann_volume_name = 'annotation'			
    ann_volume_mount_path = '/annotations'	
    ratio_val = 0.1
	
    _lebelme_op = lebelme_op(ann_pvc_name, ann_volume_name, ann_volume_mount_path,
                             ratio_val)
    
    _train_op = train_op(_lebelme_op.outputs['train_dataset'] )

    
    _train_op.after(_lebelme_op)
    
    
if __name__=="__main__":
    kfp.compiler.Compiler().compile(ITC_pipeline, "./ITC_pipeline.yaml")