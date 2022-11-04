"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from runner_utils.mapping import map_bbox_to_image_coords, map_keypoint_to_image_coords

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)       # opencv loads file in BGR format
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
THRESHOLD = 0.6               # ignore keypoints below this threshold
KP_LEFT_ANKLE = 15         # PoseNet's skeletal keypoints
KP_RIGHT_ANKLE = 16

def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.4,
      color=color_code,
      thickness=2,
   )

class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.stride = None
        # setup object working variables

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node draws keypoints.

        Args:
            inputs (dict): Dictionary with keys 
                "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

        Returns:
            outputs (dict): Dictionary with one key
                "stride"
        """
        
        # get required inputs from pipeline
        img = inputs["img"]
        bboxes = inputs["bboxes"]
        bbox_scores = inputs["bbox_scores"]
        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]

        img_size = (img.shape[1], img.shape[0])  # image width, height

        # get bounding box confidence score and draw it at the
        # bottom of the bounding box
        try: 
            the_bbox = bboxes[0]             # image only has one person
            the_bbox_score = bbox_scores[0]
            
            x1, y1, x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
        
            score_str = f"BBox {the_bbox_score:0.2f}"
            cv2.putText(
                img=img,
                text=score_str,
                org=(x1-50, y1+50),  
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=WHITE,
                thickness=1,
            )
            
            the_keypoints = keypoints[0]              # image only has one person
            the_keypoint_scores = keypoint_scores[0]  # only one set of scores
            right_ankle = None
            left_ankle = None
            
            for i, keypoints in enumerate(the_keypoints):
                keypoint_score = the_keypoint_scores[i]
            
                if keypoint_score >= THRESHOLD:
                    x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
                    x_y_str = f"({x}, {y})"
                                    
                    the_color = WHITE
                    
                    if i == KP_LEFT_ANKLE:
                        right_ankle = keypoints
                        the_color = RED
                    if i == KP_RIGHT_ANKLE:
                        left_ankle = keypoints
                        the_color = BLUE
                    
                    draw_text(img, x, y, x_y_str, the_color)
            
            if right_ankle is not None and left_ankle is not None:
                self.stride = abs(right_ankle[0] - left_ankle[0])
                
                stride_str = f"Stride Distance = {self.stride}"
                the_color = PURPLE
                draw_text(img, 30, 30, stride_str, the_color)
        except IndexError:
            pass
        return {"stride":self.stride}