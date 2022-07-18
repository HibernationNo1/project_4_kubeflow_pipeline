import kfp
from kfp import dsl
from kfp import onprem

import os

# image, dataset.json 을 save
def lebelme_op(pvc_name, volume_name, volume_mount_path, ratio_val):
    return dsl.ContainerOp(
        name='lebelme',
        image='hibernation4958/labelme:0.2',
        arguments=['--ann_path', volume_mount_path,
                   '--ratio-val', ratio_val],
        file_outputs = {'train_dataset' : '/train_dataset/train_dataset.json'}
    	).apply(
        onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def train_op(dataset_path):
    return dsl.ContainerOp(
        name='train',
        image='hibernation4958/train:0.2', 
        arguments=['--dataset_path', dataset_path]
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
    
    print(f"type(_lebelme_op.outputs) : {type(_lebelme_op.outputs)}")
    print(f"keys : {_lebelme_op.outputs.keys()}")
    print(f"_lebelme_op.outputs['train_dataset'] : {_lebelme_op.outputs['train_dataset']}")
    
    _train_op = train_op(_lebelme_op.outputs['train_dataset']).after(_lebelme_op)

    print(f"_train_op : {_train_op},            done??")

    
    
if __name__=="__main__":
    kfp.compiler.Compiler().compile(ITC_pipeline, "./ITC_pipeline.yaml")