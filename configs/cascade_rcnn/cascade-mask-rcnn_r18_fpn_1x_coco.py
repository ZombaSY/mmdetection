_base_ = [
    '../_base_/models/cascade-mask-rcnn_r18_fpn-hairseg.py',
    '../_base_/datasets/coco_instance-hairseg.py',
    '../_base_/schedules/schedule_1x-hairseg.py', '../_base_/default_runtime-hairseg.py'
]
