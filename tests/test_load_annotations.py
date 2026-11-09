def test_coco_load_annotations(mini_coco):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_materialize import Instance
    ds = CocoSegmentationDataset(root=mini_coco, split="train", layout="coco")
    inst = ds._load_annotations(0)
    assert len(inst) == 2
    assert all(isinstance(i, Instance) for i in inst)
    assert {i.label for i in inst} == {0, 1}
    assert all(i.polygon is not None for i in inst)  # mini_coco uses polygon seg
    assert ds._image_path(0).endswith(".jpg")
