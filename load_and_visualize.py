import cv2
import os
import numpy
import time
import json
import sys
from concurrent import futures


VISUALIZE_MOCAP = False
DATA_ROOT = "./"

SKIP = 2

IMAGE_FOLDERS = [
    "images/cam_rgb/",
    "images/cam_depth/",
    "images/cam_left/",
    "images/cam_right/",
    "images/cam_fisheye/",
]
ALIGNED_IMAGE_LIST_FILE = "align_result.csv"


UP_BORDER = 8
DOWN_BORDER = 8
LEFT_BORDER = 8
RIGHT_BORDER = 8

ANNOTATION_HEIGHT = 30
BAR_HEIGHT = 20

SINGLE_IMAGE_WIDTH = 640
SINGLE_IMAGE_HEIGHT = 360

FULL_IMAGE_WIDTH = (LEFT_BORDER + SINGLE_IMAGE_WIDTH + RIGHT_BORDER) * 3
FULL_IMAGE_HEIGHT = (
    (UP_BORDER + SINGLE_IMAGE_HEIGHT + DOWN_BORDER) * 2 + ANNOTATION_HEIGHT + BAR_HEIGHT
)

IMAGE_COORD = [
    (
        (LEFT_BORDER + SINGLE_IMAGE_WIDTH + RIGHT_BORDER) + LEFT_BORDER,
        (UP_BORDER + SINGLE_IMAGE_HEIGHT + DOWN_BORDER) + UP_BORDER,
    ),
    (
        (LEFT_BORDER + SINGLE_IMAGE_WIDTH // 2)
        + LEFT_BORDER
        + SINGLE_IMAGE_WIDTH
        + RIGHT_BORDER,
        UP_BORDER,
    ),
    (LEFT_BORDER, (UP_BORDER + SINGLE_IMAGE_HEIGHT + DOWN_BORDER) + UP_BORDER),
    (
        (SINGLE_IMAGE_WIDTH + LEFT_BORDER + RIGHT_BORDER) * 2 + LEFT_BORDER,
        (UP_BORDER + SINGLE_IMAGE_HEIGHT + DOWN_BORDER) + UP_BORDER,
    ),
    ((LEFT_BORDER + SINGLE_IMAGE_WIDTH // 2) + LEFT_BORDER, UP_BORDER),
]
ANNOTATION_COORD_Y = FULL_IMAGE_HEIGHT - BAR_HEIGHT - ANNOTATION_HEIGHT // 3

JOBS_NUMBER = 5

FOOT_IMAGE_WIDTH = SINGLE_IMAGE_WIDTH * 3 // 16
FOOT_IMAGE_HEIGHT = SINGLE_IMAGE_HEIGHT * 2 // 3
FOOT_COORD = {
    "left_foot": (
        UP_BORDER + FOOT_IMAGE_HEIGHT // 2,
        SINGLE_IMAGE_WIDTH * 5 // 16,
    ),
    "right_foot": (
        UP_BORDER + FOOT_IMAGE_HEIGHT // 2,
        (LEFT_BORDER + SINGLE_IMAGE_WIDTH // 2)
        + (LEFT_BORDER + SINGLE_IMAGE_WIDTH + RIGHT_BORDER) * 2,
    ),
}

HAND_IMAGE_WIDTH = SINGLE_IMAGE_WIDTH * 3 // 16
HAND_IMAGE_HEIGHT = SINGLE_IMAGE_HEIGHT // 3
HAND_COORD = {
    "left_hand": (
        UP_BORDER,
        SINGLE_IMAGE_WIDTH * 5 // 16,
    ),
    "right_hand": (
        UP_BORDER,
        (LEFT_BORDER + SINGLE_IMAGE_WIDTH // 2)
        + (LEFT_BORDER + SINGLE_IMAGE_WIDTH + RIGHT_BORDER) * 2,
    ),
}


def depth_to_rgb(depth_image):
    MIN_VALUE = 0.0
    MAX_VALUE = 10.0
    clipped_depth = numpy.clip(depth_image, MIN_VALUE, MAX_VALUE)
    # Normalize the clipped depth image
    normalized_depth = ((clipped_depth - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)) * 255
    normalized_depth = normalized_depth.astype(numpy.uint8)

    # Apply a colormap to the normalized depth image
    colormap = cv2.COLORMAP_JET
    rgb_image = cv2.applyColorMap(normalized_depth, colormap)

    return rgb_image


def draw_image(full_image, image_path, width, height, ox, oy, depth=False, index=0):
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        return None, index
    image = cv2.imread(image_path)
    try:
        s = image.shape
    except:
        print(image_path)
        exit(2)

    if depth:
        image = depth_to_rgb(image)

    image = cv2.resize(image, (width, height))

    full_image[oy : oy + height, ox : ox + width, :] = image
    return image, index


def load_aligned_image_list(aligned_image_list_file):
    aligned_image_list = []
    with open(aligned_image_list_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            image_list = line.split(",")
            aligned_image_list.append(image_list[1:])
    return aligned_image_list


def matrix_to_foot_image(matrix, width=200, height=100, dot_radius=5, left=False):
    matrix = numpy.array(matrix).reshape((15, 7))
    matrix_l, matrix_w = matrix.shape
    image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    max_dot_radius = min(
        numpy.floor(width / matrix_l / 2), numpy.floor(height / matrix_w / 2)
    )
    dot_radius = min(dot_radius, max_dot_radius)
    x_offset = (width - matrix_l * max_dot_radius * 2) // 2
    y_offset = (height - matrix_w * max_dot_radius * 2) // 2

    def draw_circle(x, y, v):
        MIN_V, MAX_V = 0, 4095
        if not MIN_V <= v <= MAX_V:
            raise Exception(f"invalid v: {v}, should be between {MIN_V} and {MAX_V}")

        def decide_color(v):
            l, u = 2000, 4000
            if v > u:
                return (128, 128, 128)
            if v < l:
                return (0, 0, 255)

            l_color, u_color = (0, 0, 255), (0, 255, 255)
            ratio = (v - l) / (u - l)
            return tuple(
                int(l_c + ratio * (u_c - l_c)) for l_c, u_c in zip(l_color, u_color)
            )

        cv2.circle(image, (x, y), dot_radius, decide_color(v), -1)

    for i in range(matrix_l):
        for j in range(matrix_w):
            draw_circle(
                x=int(i * max_dot_radius * 2 + max_dot_radius + x_offset),
                y=int(j * max_dot_radius * 2 + max_dot_radius + y_offset),
                v=matrix[i, j],
            )

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if left:
        image = cv2.flip(image, 1)
    return image


def array_to_hand_image(array, width=180, height=180, dot_radius=7, left=False):
    matrix = numpy.ones((12, 12), dtype=numpy.uint16) * 65535
    matrix_l, matrix_w = matrix.shape
    matrix[5, 11] = array[0]  # thumb
    matrix[7, 9] = array[1]  # thumb
    matrix[5, 8] = array[2]  # index
    matrix[3, 9] = 0
    matrix[1, 10] = array[3]  # index
    matrix[0, 7] = array[4]  # middle
    matrix[3, 7] = 0
    matrix[5, 7] = array[5]  # middle
    matrix[0, 5] = array[6]  # ring
    matrix[3, 5] = 0
    matrix[5, 5] = array[7]  # ring
    matrix[1, 2] = array[8]  # pinky
    matrix[3, 3] = 0
    matrix[5, 4] = array[9]  # pinky
    matrix[8, 5] = array[10]
    matrix[8, 7] = array[11]
    matrix[9, 6] = array[12]
    max_dot_radius = min(
        numpy.floor(width / matrix_l / 2), numpy.floor(height / matrix_w / 2)
    )
    dot_radius = min(dot_radius, max_dot_radius)
    x_offset = (width - matrix_l * max_dot_radius * 2) // 2
    y_offset = (height - matrix_w * max_dot_radius * 2) // 2
    image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    def draw_circle(x, y, v):
        MIN_V, MAX_V = 0, 65535
        if not MIN_V <= v <= MAX_V:
            raise Exception(f"invalid v: {v}, should be between {MIN_V} and {MAX_V}")

        def decide_color(v):
            if v == 65535:
                return (0, 0, 0)
            l, u = 1, 6000
            if v > u:
                return (255, 0, 0)
            if v < l:
                return (128, 128, 128)

            l_color, u_color = (255, 0, 0), (255, 255, 0)
            ratio = (v - l) / (u - l)
            return tuple(
                int(l_c + ratio * (u_c - l_c)) for l_c, u_c in zip(l_color, u_color)
            )

        cv2.circle(image, (x, y), dot_radius, decide_color(v), -1)

    for i in range(matrix_l):
        for j in range(matrix_w):
            draw_circle(
                x=int(i * max_dot_radius * 2 + max_dot_radius + x_offset),
                y=int(j * max_dot_radius * 2 + max_dot_radius + y_offset),
                v=matrix[i, j],
            )
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if left:
        image = cv2.flip(image, 1)
    return image


def load_haptics_data(haptics_data_path):
    data = []
    with open(haptics_data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            touch_data = line.strip().split(",")[2:]
            data.append([int(x) for x in touch_data])
    return data


def load_mocap_data(mocap_data_path):
    data = []
    with open(mocap_data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            line = line.strip().split(",")
            data.append([float(x) for x in line])
    return data


def load_annotation_data(annotation_data_path):
    annotations = []
    with open(annotation_data_path, "r") as f:
        data = json.load(f)
        for annotation in data["subtasks"]:
            annotations.append(annotation)
    return annotations


class Human:
    def __init__(self):
        urdf_path = os.path.join(os.path.dirname(__file__), "human.urdf")
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.joint_indices = range(p.getNumJoints(self.robot))
        self.joint_mapping = {
            "base": "joint_Hips",
            "dorsal": "joint_Spine",
            "head": "joint_Head",
            "l_arm": "joint_LeftArm",
            "l_forarm": "joint_LeftForeArm",
            "l_hand": "joint_LeftHand",
            "r_arm": "joint_RightArm",
            "r_forarm": "joint_RightForeArm",
            "r_hand": "joint_RightHand",
            "l_thigh": "joint_LeftUpLeg",
            "l_leg": "joint_LeftLeg",
            "l_foot": "joint_LeftFoot",
            "r_thigh": "joint_RightUpLeg",
            "r_leg": "joint_RightLeg",
            "r_foot": "joint_RightFoot",
        }
        self.motion_quat = [(0, 0, 0, 1)] * p.getNumJoints(self.robot)

    def sync_data(self, motion_quat, name):
        motion_quat = motion_quat[1:] + [motion_quat[0]]
        for joint_index in self.joint_indices:
            joint_name = p.getJointInfo(self.robot, joint_index)[1].decode("utf8")
            if name in self.joint_mapping and joint_name == self.joint_mapping[name]:
                # if name == "mocap_l_hand":
                # print(joint_index, joint_name, motion_quat)
                p.resetJointStateMultiDof(self.robot, joint_index, motion_quat)
                break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_and_visualize.py <data_path>")
        exit(1)
    DATA_ROOT = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--visualize_mocap":
        VISUALIZE_MOCAP = True

    if VISUALIZE_MOCAP:
        import pybullet as p
        import pybullet_data

    aligned_image_list = load_aligned_image_list(
        os.path.join(DATA_ROOT, ALIGNED_IMAGE_LIST_FILE)
    )
    foot_haptics_data = {
        "left_foot": [],
        "right_foot": [],
    }
    hand_haptics_data = {
        "left_hand": [],
        "right_hand": [],
    }
    mocap_data = {
        "base": [],
        "dorsal": [],
        "head": [],
        "l_arm": [],
        "l_forarm": [],
        "l_hand": [],
        "l_foot": [],
        "l_leg": [],
        "l_thigh": [],
        "l_thumb_0": [],
        "l_thumb_1": [],
        "l_index_0": [],
        "l_index_1": [],
        "l_middle_0": [],
        "l_middle_1": [],
        "l_pinky_0": [],
        "l_pinky_1": [],
        "l_ring_0": [],
        "l_ring_1": [],
        "r_foot": [],
        "r_forarm": [],
        "r_hand": [],
        "r_leg": [],
        "r_arm": [],
        "r_middle_0": [],
        "r_middle_1": [],
        "r_pinky_0": [],
        "r_pinky_1": [],
        "r_ring_0": [],
        "r_ring_1": [],
        "r_thigh": [],
        "r_thumb_0": [],
        "r_thumb_1": [],
        "r_index_0": [],
        "r_index_1": [],
    }

    for k in foot_haptics_data:
        foot_haptics_data[k] = load_haptics_data(
            os.path.join(DATA_ROOT, f"haptics_aligned/{k}_aligned.csv")
        )
    for k in hand_haptics_data:
        hand_haptics_data[k] = load_haptics_data(
            os.path.join(DATA_ROOT, f"haptics_aligned/{k}_aligned.csv")
        )
    sequence_length = min(
        [len(aligned_image_list)]
        + [len(hand_haptics_data[k]) for k in hand_haptics_data]
        + [len(foot_haptics_data[k]) for k in foot_haptics_data]
    )
    if VISUALIZE_MOCAP:
        for k in mocap_data:
            mocap_data[k] = load_mocap_data(
                os.path.join(DATA_ROOT, f"motion_capture_aligned/{k}_aligned.csv")
            )
        sequence_length = min(
            [sequence_length] + [len(mocap_data[k]) for k in mocap_data]
        )
    cv2.namedWindow("Visualizer", cv2.WINDOW_NORMAL)

    data_index = 0

    def on_change(emp):
        global data_index
        data_index = emp if emp <= sequence_length - 1 else sequence_length - 1

    cv2.createTrackbar("trackbar", "Visualizer", 0, sequence_length, on_change)

    # import pybullet

    image = numpy.zeros(
        (FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH, 3),
        dtype=numpy.uint8,
    )
    bar = numpy.zeros((BAR_HEIGHT, FULL_IMAGE_WIDTH, 3), dtype=numpy.uint8)
    annotations = load_annotation_data(os.path.join(DATA_ROOT, "annotation.json"))
    for annotation in annotations:
        start_x = int(
            numpy.ceil(
                int(annotation["start_frame_id"]) / sequence_length * FULL_IMAGE_WIDTH
            )
        )
        end_x = int(
            numpy.floor(
                int(annotation["end_frame_id"]) / sequence_length * FULL_IMAGE_WIDTH
            )
        )
        if end_x >= FULL_IMAGE_WIDTH - 1:
            end_x = FULL_IMAGE_WIDTH - 1
        bar[:, start_x:end_x, 2:].fill(255)

    if VISUALIZE_MOCAP:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, 0)
        human = Human()

    while True:
        t0 = time.time()
        with futures.ThreadPoolExecutor(max_workers=JOBS_NUMBER) as executor:
            to_do = []
            for i in range(JOBS_NUMBER):
                image_path = os.path.join(
                    DATA_ROOT, IMAGE_FOLDERS[i] + aligned_image_list[data_index][i]
                )
                is_depth = i == 1

                to_do.append(
                    executor.submit(
                        draw_image,
                        image,
                        image_path,
                        SINGLE_IMAGE_WIDTH,
                        SINGLE_IMAGE_HEIGHT,
                        IMAGE_COORD[i][0],
                        IMAGE_COORD[i][1],
                        is_depth,
                        i,
                    )
                )

            result = [None] * JOBS_NUMBER
            for future in futures.as_completed(to_do):
                res, index = future.result()
                result[index] = res
        for k in foot_haptics_data:
            foot_image = matrix_to_foot_image(
                foot_haptics_data[k][data_index], left="left" in k
            )
            foot_image = cv2.resize(foot_image, (FOOT_IMAGE_WIDTH, FOOT_IMAGE_HEIGHT))
            image[
                FOOT_COORD[k][0] : FOOT_COORD[k][0] + FOOT_IMAGE_HEIGHT,
                FOOT_COORD[k][1] : FOOT_COORD[k][1] + FOOT_IMAGE_WIDTH,
                :,
            ] = foot_image
        for k in hand_haptics_data:
            hand_image = array_to_hand_image(
                hand_haptics_data[k][data_index], left="left" in k
            )
            hand_image = cv2.resize(hand_image, (HAND_IMAGE_WIDTH, HAND_IMAGE_HEIGHT))
            image[
                HAND_COORD[k][0] : HAND_COORD[k][0] + HAND_IMAGE_HEIGHT,
                HAND_COORD[k][1] : HAND_COORD[k][1] + HAND_IMAGE_WIDTH,
                :,
            ] = hand_image
        if VISUALIZE_MOCAP:
            for k in mocap_data:
                human.sync_data(mocap_data[k][data_index][2:6], k)

        image[-BAR_HEIGHT:, :, :] = bar
        bar_box_pos = int(data_index / sequence_length * FULL_IMAGE_WIDTH)
        box_width = 2
        if bar_box_pos >= FULL_IMAGE_WIDTH - 1 - box_width:
            bar_box_pos = FULL_IMAGE_WIDTH - 1 - box_width
        image[-BAR_HEIGHT:, bar_box_pos : bar_box_pos + box_width, :] = (0, 255, 0)

        image[-ANNOTATION_HEIGHT - BAR_HEIGHT : -BAR_HEIGHT, :, :].fill(0)
        for annotation in annotations:
            if (
                int(annotation["start_frame_id"])
                <= data_index
                <= int(annotation["end_frame_id"])
            ):
                cv2.addText(
                    image,
                    annotation["description"],
                    (20, ANNOTATION_COORD_Y),
                    "Arial",
                    int(ANNOTATION_HEIGHT * 0.6),
                    (0, 128, 255),
                    50,
                    cv2.LINE_8,
                )
                break
        cv2.imshow("Visualizer", image)
        cv2.setTrackbarPos("trackbar", "Visualizer", data_index)
        data_index += SKIP
        if data_index >= sequence_length - 1:
            data_index = sequence_length - 1
        t1 = time.time()
        loading_time_ms = int(round((t1 - t0) * 1000))
        waitkey_time = 33 - loading_time_ms if loading_time_ms < 33 else 1
        if cv2.waitKey(waitkey_time) & 0xFF == ord("q"):
            break
