"""
scene_understanding — Scene Segmentation Module
================================================
Scene image → instance segmentation masks.

Currently uses Mask2Former (panoptic) via mask2former_gaze_collection.py.
PSPNet (semantic) is available as a fallback via mmseg.

Target interface:
    model = SceneSegmenter(model_type="mask2former")
    instances = model.predict(scene_image)  # -> list of InstanceMask
"""
