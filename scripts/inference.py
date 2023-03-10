__doc__ = """
This module is a single class for image segmentation using a trained YOLOv5 model.
"""


from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray
from PIL import Image

SEGMENTER_MODEL_PATH = "models/best.onnx"
IMAGE_INPUT_WIDTH = 640
IMAGE_INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.1
SCORES_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

Number = Union[int, float]
NumberTuple = Tuple[Union[int, float], Union[int, float]]


class LayoutSegmenter:
    """Layout segmentation"""

    __yolov5_model: Any = None

    @classmethod
    def load_yolov5_model(cls) -> Any:
        """load the resume layout segmentation model

        Returns:
            Any: Yolov5 trained model
        """
        if not cls.__yolov5_model:
            # pylint: disable=no-member
            cls.__yolov5_model = cv2.dnn.readNetFromONNX(SEGMENTER_MODEL_PATH)
        return cls.__yolov5_model

    def segment(self, page_image: Image.Image) -> ndarray:
        """Predict the existing sections in an image and get their coordonates

        Returns:
            List[List[float]]: List of results or sections coordonates each
                has xmin, ymin, xmax, ymax
        """
        preprocessed_image = self.preprocess_image(page_image)
        predictions = self.get_predictions(preprocessed_image)
        predicted_boxes = self.preprocess_predictions(
            preprocessed_image, predictions[0]
        )
        if predicted_boxes.size:
            predicted_boxes = self.xywh_to_xyxy(predicted_boxes)
        return predicted_boxes

    def get_predictions(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Get predictions using opencv and the trained YOLOv5 model.

        Args:
            preprocessed_image (np.ndarray): Preprocesed image to match the model
                input format.

        Returns:
            np.ndarray: Model predictions and confidences.
        """
        # pylint: disable=no-member
        blob = cv2.dnn.blobFromImage(
            preprocessed_image,
            1 / 255.0,
            (IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT),
            swapRB=True,
            crop=False,
        )
        segmentation_model = self.load_yolov5_model()
        segmentation_model.setInput(blob)
        predictions = segmentation_model.forward()
        return predictions

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image to match YOLOv5 input format.

        Args:
            image (Image.Image): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        array_image = np.asanyarray(image)
        array_image = self.preprocess_color(array_image)
        row, col, _ = array_image.shape
        _max = max(col, row)
        preprocessed_image = np.zeros((_max, _max, 3), np.uint8)
        preprocessed_image[0:row, 0:col] = array_image
        return preprocessed_image

    def preprocess_predictions(
        self, input_image: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Preprocess predictions to remove predictions with low confidence
        and remove overlaped ones.

        Args:
            input_image (np.ndarray): Model input image.
            predictions (np.ndarray): Model predictions.

        Returns:
            np.ndarray: The preprocessed predictions.
        """
        confidences = []
        boxes = []
        rows = predictions.shape[0]
        x_scale, y_scale = self.get_scales(
            input_image, IMAGE_INPUT_WIDTH, IMAGE_INPUT_WIDTH
        )
        for r_idx in range(rows):
            row = predictions[r_idx]
            confidence = row[4]
            if confidence >= CONFIDENCE_THRESHOLD:
                confidences.append(confidence)
                boxes.append(self.xyxy_to_xywh(row, x_scale, y_scale))
        result_boxes = self.remove_overlaped_boxes(boxes, confidences)
        return result_boxes

    def xyxy_to_xywh(
        self, row: np.ndarray, x_scale: Number, y_scale: Number
    ) -> np.ndarray:
        """Convert coordonates xmin, ymin, xmax, ymax to xmin, ymin, width, height formats.

        Args:
            row (np.ndarray): Array containing xmin, ymin, xmax, ymax.
            x_scale (Number): The x scale between the original image and the
                YOLOv5 input image sizes.
            y_scale (Number): The y scale between the original image and the
                YOLOv5 input image sizes.

        Returns:
            np.ndarray: The converted coordonates
        """
        x_coord, y_coord, width_, height_ = (
            row[0].item(),
            row[1].item(),
            row[2].item(),
            row[3].item(),
        )
        left = int((x_coord - 0.5 * width_) * x_scale)
        top = int((y_coord - 0.5 * height_) * y_scale)
        width = int(width_ * x_scale)
        height = int(height_ * y_scale)
        return np.array([left, top, width, height])

    def remove_overlaped_boxes(
        self, boxes: List[np.ndarray], confidences: List[Number]
    ) -> np.ndarray:
        """Remove the overlaped boxes using opencv dnn.NMSBoxes
        based on `SCORES_THRESHOLD` and `NMS_THRESHOLD` values.

        `SCORES_THRESHOLD`: a threshold used to filter boxes by score.
        `NMS_THRESHOLD`: a threshold used in non maximum suppression.

        Args:
            boxes (List[np.ndarray]): Input boxes array.
            confidences (List[Number): List of boxes confidences.

        Returns:
            np.ndarray: Non overlaped boxes array.
        """
        # pylint: disable=no-member
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORES_THRESHOLD, NMS_THRESHOLD)
        return np.asarray([boxes[idx] for idx in indexes])

    def xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes xmin, ymin, width, height format to
            xmin, ymin, xmax, ymax formats.

        Args:
            boxes (np.ndarray): Input boxes array.

        Returns:
            np.ndarray: The converted boxes array.
        """
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return boxes

    def preprocess_color(self, image: np.ndarray) -> np.ndarray:
        """Precess image color by converting from different image color formats.

        Args:
            image (np.ndarray): Input image array to process.

        Returns:
            np.ndarray: The processed image array.
        """
        preprocessed_image = self.bgr_to_gray(image)
        preprocessed_image = self.gray_to_bgr(preprocessed_image)
        return preprocessed_image

    def gray_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """Convert image color from GRAY to BGR"""
        # pylint: disable=no-member
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def bgr_to_gray(self, gray_image: np.ndarray):
        """Convert image color from BGR to GRAY"""
        # pylint: disable=no-member
        return cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    def get_scales(
        self,
        input_image: np.ndarray,
        width: Number,
        height: Number,
    ) -> NumberTuple:
        """Get x and y scales between the input image shape and the `width` and the `height`

        Args:
            input_image (np.ndarray): Input image array.
            width (Number): Other image width.
            height (Number): Other image height.

        Returns:
            NumberTuple: X and Y scales tuple.
        """
        image_width, image_height, _ = input_image.shape
        x_scale = image_width / width
        y_scale = image_height / height
        return (x_scale, y_scale)


T_Num = Union[int, float]
T_TupleNum = Tuple[T_Num, T_Num]


T_BBoxes = ndarray
T_Overlaped_BBoxes = List[Tuple[int, T_BBoxes]]

def filter_bboxes(bboxes: T_BBoxes) -> T_BBoxes:
    """Filter overlaped bounding boxes/coordonates

    Args:
        bboxes (T_BBoxes): Input bouding boxes

    Returns:
        T_BBoxes: The filtered bounding boxes
    """
    n_dels = 0
    try:
        sorted_bboxes = bboxes[bboxes[:, 3].argsort()]
        intersected = get_intersected_bbox_indexes(sorted_bboxes)
        for index1, index2 in intersected:
            index1 = index1 - n_dels
            index2 = index2 - n_dels
            area1 = calculate_area(sorted_bboxes[index1])
            area2 = calculate_area(sorted_bboxes[index2])
            del_idx = index2 if area1 > area2 else index1
            sorted_bboxes = np.delete(sorted_bboxes, del_idx, 0)
            n_dels += 1
        return sorted_bboxes
    except IndexError:
        return bboxes


def get_intersected_bbox_indexes(bboxes: T_BBoxes) -> T_Overlaped_BBoxes:
    """Get the indexes of the overlaped bounding boxes

    Args:
        bboxes (T_BBoxes): Input bounding boxes

    Returns:
        T_Overlaped_BBoxes: List of the overlaped tuples
    """
    intersected_indexes = []
    for index, item in enumerate(bboxes):
        for sub_index, sub_item in enumerate(bboxes):
            if index != sub_index:
                pr_inter = calculate_intersection_percentage(item, sub_item)
                if pr_inter > 0.5:
                    if (sub_index, index) not in intersected_indexes:
                        intersected_indexes.append((index, sub_index))
    return intersected_indexes


def calculate_intersection_percentage(
    rectangle1: T_BBoxes, rectangle2: T_BBoxes
) -> float:
    """Get the overlaping pourcentage between two rectangles

    Args:
        rectangle1 (T_BBoxes): The first rectangle coordonates
        rectangle2 (T_BBoxes): The second rectangle coordonates

    Returns:
        float: The overlaping pourcentage
    """
    inter_area = max(
        0, min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0])
    ) * max(0, min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1]))
    rectangle_area1 = calculate_area(rectangle1)
    rectangle_area2 = calculate_area(rectangle2)
    return (
        inter_area / rectangle_area2
        if rectangle_area1 > rectangle_area2
        else inter_area / rectangle_area1
    )


def calculate_area(rectangle: T_BBoxes) -> T_Num:
    """Calculate the area of a rectangle

    Args:
        rectangle (T_BBoxes): The rectangle coordonates

    Returns:
        float: The calculated area height * width
    """
    width = rectangle[2] - rectangle[0]
    height = rectangle[3] - rectangle[1]
    return width * height


def get_scales(input_image: np.ndarray, width: T_Num, height: T_Num) -> T_TupleNum:
    """Get x and y scales between the input image shape and the `width` and the `height`

    Args:
        input_image (np.ndarray): Input image array.
        width (T_Num): Other image width.
        height (T_Num): Other image height.

    Returns:
        T_TupleNum: X and Y scales tuple.
    """
    image_width, image_height, _ = input_image.shape
    x_scale = image_width / width
    y_scale = image_height / height
    return (x_scale, y_scale)
