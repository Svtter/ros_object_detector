#!/usr/bin/env python
# coding: utf-8
"""
ROS与Tensorflow结合模块

"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from geometry_msgs.msg import Point
from object_detector.msg import Detect, Box

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from multiprocessing import Queue, Pool
import tensorflow as tf
#from utils.app_utils import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class RosTensorFlow():
    """
    ROS与Tensorflow结合模块

    1. 需要从外部载入detector_graph

    """

    def __init__(self):
        self._cv_bridge = CvBridge()
        # self._sub = rospy.Subscriber('/kinect2/qhd/image_color', Image,
        # self.callback, queue_size=1, buff_size=2**24)

        # 订阅kinect2
        # TODO: 指定topic
        self._sub = rospy.Subscriber('/kinect2/qhd/image_color', Image, self.callback, queue_size=1, buff_size=2**24)
        rospy.loginfo('Image converter constructor ')

        self._pub = rospy.Publisher("/image_objects_detect", Image, queue_size=0)

        # 发布盒信息
        self._pub_box = rospy.Publisher("/image_objects_detect/box", Detect, queue_size=10)
        self._pub_center = rospy.Publisher("/image_objects_detect/center", Point, queue_size=10)

        self._detect = Detect()

        #self._pub = rospy.Publisher('result', Int16, queue_size=1)

    def detect_objects(image_np, sess, detection_graph):
        """
        检测物体

        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image_np

    def callback(self, image_msg):
        """
        回调函数
        """

        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        boxes, scores, classes = None, None, None

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_np = cv_image
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes)


                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

        try:
            self._pub.publish(self._cv_bridge.cv2_to_imgmsg(image_np, "bgr8"))

            y, x, _ = cv_image.shape

            # self._detect.boxes = map(Box, boxes)
            # self._detect.boxes = boxes
            centers = [ [(box[1] + box[3])/2 * x, (box[0] + box[2])/2 * y] for box in boxes]

            p_boxes = []
            for center in centers:
                temp = Box()
                temp.box = center
                p_boxes.append(temp)

            # print('p_boxes is: ', p_boxes)

            self._detect.boxes = p_boxes
            self._detect.scores = scores
            self._detect.classes = classes

            # show detect box
            rospy.loginfo(self._detect.boxes[:3])
            rospy.loginfo(self._detect.classes[:3])

            detect = [category_index[c] for c in classes[:3]]

            # print image_np
            # print('shape: ', y, x)

            # output msg info
            # rospy.loginfo(detect)

            # 求中心点
            # center = [ [(box[1] + box[3])/2 * x, (box[0] + box[2])/2 * y] for box in boxes]

            # rospy.loginfo('scores: ')
            # rospy.loginfo(scores[:3])
            # rospy.loginfo('center: ')
            # rospy.loginfo(center[:3])

            self._pub_box.publish(self._detect)

        except CvBridgeError as e:
            ros.logerror(e)


    def main(self):
        """
        ROS主类，负责调用ROS-spin循环
        """
        rospy.spin()


def get_package_path():
    """
    获取当前包的URL
    """

    import rospkg

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # list all packages, equivalent to rospack list
    # rospack.list_pkgs()

    # get the file path for rospy_tutorials
    PACKAGE_PATH = rospack.get_path('object_detector')

    return PACKAGE_PATH



if __name__ == '__main__':

    rospy.init_node('rostensorflow')
    #CWD_PATH = os.getcwd()

    PACKAGE_PATH = get_package_path()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
    # 选择加载的模型
    PATH_TO_CKPT = PACKAGE_PATH + '/include/obj_detector/quail_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    # 选择加载的标签
    PATH_TO_LABELS = PACKAGE_PATH + '/include/obj_detector/object-detection.pbtxt'

    # 分类的个数
    NUM_CLASSES = 90

    # detection_graph Tensorflow 检测图
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    tensor = RosTensorFlow()
    tensor.main()
