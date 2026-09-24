import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from ultralytics import YOLO
from rclpy.executors import MultiThreadedExecutor
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.optimize import linear_sum_assignment
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
import time

class CameraSub(Node):

    def __init__(self):
        super().__init__('camera_sub')

        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.camera_info = None

        self.camera_info_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.world_objects_lock = threading.Lock()

        self.camera_info = None
        self.color_camera_info = None

        self.rectify_map_x = None
        self.rectify_map_y = None
        self.rectify_signature = None

        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_profile)
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=qos_profile)
        self.color_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.color_info_callback, qos_profile=qos_profile)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_callback, qos_profile=qos_profile)

        self.synchronizer = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.08)
        self.synchronizer.registerCallback(self.synced_callback)

        self.bridge = CvBridge()
        self.model = YOLO("/models/yolo11n-seg.pt")

        self.current_objects = {}
        self.world_objects = {}
        self.next_world_object_id = 0
        self.track_to_world = {}
        self.track_last_seen_ns = {}

        self.dynamic_labels = {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "dog",
            "cat"
        }

        # World-object association parameters.
        self.dynamic_base_association_distance_m = 0.35
        self.dynamic_max_speed_mps = 2.0
        self.dynamic_max_association_distance_m = 1.75

        self.static_association_distance_m = 0.60

        # Reject unreasonable velocity measurements when updating
        # the motion estimate.
        self.maximum_velocity_update_mps = 3.0

        # Do not extrapolate velocity arbitrarily far into the future.
        self.maximum_prediction_time_s = 1.0

        # Global one-to-one association parameters.
        self.dynamic_reidentification_age_s = 10.0
        self.dynamic_active_age_s = 2.0
        # Static objects remain available for later SST queries in the map.
        self.static_active_age_s = float("inf")
        self.sst_future_observation_tolerance_s = 0.10
        self.appearance_cost_weight = 0.35
        self.track_match_bonus = 0.75
        self.label_mismatch_cost = 0.75
        self.label_change_max_appearance_distance = 0.30
        self.label_change_max_spatial_cost = 0.50
        self.invalid_association_cost = 1.0e6

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def color_info_callback(self, msg):
        with self.camera_info_lock:
            self.color_camera_info = msg

    def info_callback(self, msg):
        with self.camera_info_lock:
            self.camera_info = msg

    def synced_callback(self, color_msg, depth_msg):
        with self.processing_lock:
            self.process_synced_frame(color_msg, depth_msg)

    def process_synced_frame(self, color_msg, depth_msg):
        color_camera_info = None
        camera_info = None
        with self.camera_info_lock:
            camera_info = self.camera_info
            color_camera_info = self.color_camera_info

        if camera_info is None or color_camera_info is None:
            self.get_logger().warning("Waiting for color and depth CameraInfo")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as error:
            self.get_logger().error(f"Could not convert image: {error}")
            return

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as error:
            self.get_logger().error(f"Depth conversion failed: {error}")
            return

        try:
            depth_m = self.depth_to_meters(depth_raw, depth_msg.encoding)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        image = self.rectify_color_to_depth_grid(image=image, color_camera_info=color_camera_info, depth_camera_info=camera_info, depth_shape=depth_m.shape)
        if image is None: return

        if image.shape[:2] != depth_m.shape[:2]:
            self.get_logger().error(f"RGB and depth dimensions differ: rgb={image.shape[:2]}, depth={depth_m.shape[:2]}. Depth must be registered to RGB.")
            return

        if (camera_info.width != depth_m.shape[1] or camera_info.height != depth_m.shape[0]):
            self.get_logger().error(
                "Depth CameraInfo dimensions do not match the depth image: "
                f"CameraInfo={camera_info.width}x{camera_info.height}, "
                f"depth={depth_m.shape[1]}x{depth_m.shape[0]}"
            )
            return

        try:
            inference_start = time.perf_counter()
            results = self.model.track(
                source=image,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.35,
                iou=0.5,
                imgsz=640,
                device='cpu',
                retina_masks=True,
                verbose=False
            )

            inference_seconds = time.perf_counter() - inference_start
            self.get_logger().info(f"YOLO tracking took {inference_seconds:.3f} seconds")
        except Exception as error:
            self.get_logger().error(f"YOLO tracking failed: {error}")
            return

        timestamp_ns = int(depth_msg.header.stamp.sec) * 1_000_000_000 + int(depth_msg.header.stamp.nanosec)

        if results is None or len(results) == 0:
            self.current_objects = {}
            with self.world_objects_lock:
                self.mark_stale_world_objects(timestamp_ns)
            return

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            self.current_objects = {}
            with self.world_objects_lock:
                self.mark_stale_world_objects(timestamp_ns)
            return

        if result.masks is None:
            self.get_logger().warning("YOLO returned boxes but no masks")
            self.current_objects = {}
            with self.world_objects_lock:
                self.mark_stale_world_objects(timestamp_ns)
            return

        camera_frame = depth_msg.header.frame_id
        if camera_frame is None:
            self.get_logger().error("Depth message has an empty frame_id")
            return

        try:
            camera_to_odom = self.tf_buffer.lookup_transform(
                "odom",          # target frame
                camera_frame,    # source frame
                Time.from_msg(depth_msg.header.stamp),
                timeout=Duration(seconds=0.1)
            )
        except Exception as error:
            self.get_logger().warning(f"Could not transform {camera_frame} -> odom: {error}")
            return

        frame_objects = {}
        frame_observations = []

        detected_boxes = len(result.boxes)

        unconfirmed_tracks = 0
        missing_masks = 0
        invalid_geometry = 0
        transform_failures = 0

        for index, box in enumerate(result.boxes):
            # A tracking ID might not yet be available for an
            # unconfirmed detection.
            if box.id is None:
                track_id = None
                unconfirmed_tracks += 1
            else:
                track_id = int(box.id.item())

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            xmin, ymin, xmax, ymax = [float(value) for value in box.xyxy[0].tolist()]

            label = str(result.names[class_id])

            mask = self.get_depth_sized_mask(result=result, detection_index=index, depth_shape=depth_m.shape)
            if mask is None:
                missing_masks += 1
                continue

            geometry = self.extract_object_geometry(depth_m=depth_m, object_mask=mask, camera_info=camera_info)

            if geometry is None:
                invalid_geometry += 1
                self.get_logger().debug(f"No reliable depth for {label} track {track_id}")
                continue

            center_camera = geometry["center_camera"]
            center_odom = self.transform_point_to_odom(center_camera=center_camera, depth_msg=depth_msg, transform=camera_to_odom)

            if center_odom is None:
                transform_failures += 1
                continue

            crop = self.extract_crop(image, xmin, ymin, xmax, ymax)
            observation = {
                "detection_index": index,
                "track_id": track_id,
                "label": label,
                "class_id": class_id,
                "confidence": confidence,

                "bbox_xyxy": [xmin, ymin, xmax, ymax],

                # Boolean NumPy array or None.
                "mask": mask,
                "position_camera": center_camera.tolist(),
                "position_odom": center_odom.tolist(),
                "radius_m": float(geometry["radius_m"]),
                "position_uncertainty_m": float(geometry["depth_uncertainty_m"]),
                "valid_depth_pixels": int(geometry["valid_depth_pixels"]),

                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),

                "camera_frame": camera_frame,
                "world_frame": "odom",
                "timestamp_ns": timestamp_ns,

                # The crop can later be saved to disk.
                "crop": crop,
                "appearance_descriptor": self.extract_appearance_descriptor(crop),
            }

            frame_observations.append(observation)

        # Associate the complete frame in one operation. This guarantees that
        # two detections cannot update the same persistent world object.
        with self.world_objects_lock:
            world_object_ids = self.update_world_objects_for_frame(frame_observations, timestamp_ns)

            for observation, world_object_id in zip(frame_observations, world_object_ids):
                observation["world_object_id"] = world_object_id
                track_id = observation["track_id"]
                index = observation["detection_index"]

                frame_key = f"unconfirmed_{index}" if track_id is None else track_id
                if frame_key in frame_objects:
                    frame_key = f"{frame_key}_{index}"
                frame_objects[frame_key] = observation

                world_object = self.world_objects[world_object_id]
                self.get_logger().info(
                    f"World object {world_object_id}: "
                    f"track_id={track_id}, "
                    f"label={world_object['label']}, "
                    f"position_odom={world_object['position_odom']}, "
                    f"velocity_odom={world_object['velocity_odom']}, "
                    f"radius={world_object['radius_m']:.3f} m, "
                    f"uncertainty="
                    f"{world_object['position_uncertainty_m']:.3f} m"
                )

            world_object_count = len(self.world_objects)
            active_world_object_count = sum(1 for world_object in self.world_objects.values() if world_object["active"])

        # Replace this every frame. It represents only the
        # objects visible in the current processed image.
        self.current_objects = frame_objects

        # self.get_logger().info(f"Frame contains {len(frame_objects)} 3D objects; world map contains {world_object_count} objects")
        self.get_logger().info("Frame summary: "
            f"YOLO boxes={detected_boxes}, "
            f"3D objects={len(frame_objects)}, "
            f"unconfirmed tracks={unconfirmed_tracks}, "
            f"missing masks={missing_masks}, "
            f"invalid depth geometry={invalid_geometry}, "
            f"transform failures={transform_failures}, "
            f"world objects={world_object_count}, "
            f"active world objects={active_world_object_count}"
        )

    def rectify_color_to_depth_grid(self, image, color_camera_info, depth_camera_info, depth_shape):
        """
        Rectify the raw RGB image into the registered depth
        image's pixel grid.

        This assumes that the depth image is already aligned to
        camera_color_optical_frame.
        """

        depth_height, depth_width = depth_shape

        if (color_camera_info.header.frame_id != depth_camera_info.header.frame_id):
            self.get_logger().error(
                "Color and depth CameraInfo use different frames: "
                f"color={color_camera_info.header.frame_id}, "
                f"depth={depth_camera_info.header.frame_id}. "
                "Simple image rectification is insufficient; explicit depth-to-color registration is needed."
            )
            return None

        color_k = np.asarray(color_camera_info.k, dtype=np.float64).reshape(3, 3)
        color_d = np.asarray(color_camera_info.d, dtype=np.float64)
        color_r = np.asarray(color_camera_info.r, dtype=np.float64).reshape(3, 3)

        # The depth image is the desired target pixel grid.
        # For a rectified image, P's left 3x3 block contains
        # the target projection/intrinsic matrix.
        depth_p = np.asarray(depth_camera_info.p, dtype=np.float64).reshape(3, 4)
        target_k = depth_p[:, :3]

        # Fall back to K if P is missing or invalid.
        if (abs(target_k[0, 0]) < 1e-9 or abs(target_k[1, 1]) < 1e-9):
            target_k = np.asarray(depth_camera_info.k, dtype=np.float64).reshape(3, 3)

        signature = (tuple(color_camera_info.k), tuple(color_camera_info.d), tuple(color_camera_info.r), tuple(depth_camera_info.p), depth_width, depth_height)

        if self.rectify_map_x is None or self.rectify_map_y is None or self.rectify_signature != signature:
            self.rectify_map_x, self.rectify_map_y = cv2.initUndistortRectifyMap(
                cameraMatrix=color_k,
                distCoeffs=color_d,
                R=color_r,
                newCameraMatrix=target_k,
                size=(depth_width, depth_height),
                m1type=cv2.CV_32FC1,
            )

            self.rectify_signature = signature

            self.get_logger().info(f"Created RGB rectification maps: {depth_width}x{depth_height}")

        rectified_image = cv2.remap(image, self.rectify_map_x, self.rectify_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return rectified_image

    def transform_point_to_odom(self, center_camera, depth_msg, transform):
        point_camera = PointStamped()

        point_camera.header.stamp = depth_msg.header.stamp
        point_camera.header.frame_id = depth_msg.header.frame_id

        point_camera.point.x = float(center_camera[0])
        point_camera.point.y = float(center_camera[1])
        point_camera.point.z = float(center_camera[2])

        try:
            point_odom = do_transform_point(point_camera, transform)
        except Exception as error:
            self.get_logger().warning(f"Point transformation failed: {error}")
            return None

        return np.array([point_odom.point.x, point_odom.point.y, point_odom.point.z])

    def get_depth_sized_mask(self, result, detection_index, depth_shape):
        """
        Return a Boolean segmentation mask matching the depth
        image dimensions.

        depth_shape must be (height, width).
        """

        if result.masks is None: return None
        if detection_index >= len(result.masks.data): return None

        mask = (result.masks.data[detection_index].detach().cpu().numpy())
        depth_height, depth_width = depth_shape

        if mask.shape != (depth_height, depth_width):
            mask = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)

        return mask > 0.5

    def deproject_depth_pixels(self, depth_m, valid_mask, camera_info):
        """
        Convert valid depth pixels into XYZ points in the
        depth camera optical frame.
        """

        rows, columns = np.nonzero(valid_mask)

        if len(rows) < 30: return None

        z = depth_m[rows, columns].astype(np.float64)
        u = columns.astype(np.float64)
        v = rows.astype(np.float64)

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error("CameraInfo contains invalid focal lengths")
            return None

        x = ((u - cx) * z / fx)
        y = ((v - cy) * z / fy)

        return np.column_stack((x, y, z))

    def extract_object_geometry(self, depth_m, object_mask, camera_info):
        """
        Reconstruct one segmented object in the camera frame.

        depth_m:
            HxW NumPy array containing depth in metres.

        object_mask:
            HxW Boolean NumPy array.

        camera_info:
            sensor_msgs/msg/CameraInfo.

        Returns:
            Dictionary containing center_camera, radius_m,
            depth_uncertainty_m and valid_depth_pixels.
            Returns None if depth is insufficient.
        """

        if depth_m.shape != object_mask.shape:
            self.get_logger().warning(f"Depth and object mask dimensions differ: depth={depth_m.shape}, mask={object_mask.shape}")
            return None

        # Remove mask-boundary pixels. Boundary pixels often
        # contain depth from the background.
        kernel = np.ones((5, 5), dtype=np.uint8)
        inner_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Initial depth validity test.
        valid_mask = (inner_mask & np.isfinite(depth_m) & (depth_m > 0.2)& (depth_m < 10.0))
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count < 30: return None

        depth_values = depth_m[valid_mask].astype(np.float64)
        median_depth = float(np.median(depth_values))

        # Median absolute deviation provides a robust estimate
        # of the depth spread.
        mad = float(np.median(np.abs(depth_values - median_depth)))
        robust_sigma = 1.4826 * mad

        # Keep at least a 10 cm interval because some objects
        # have very small MAD values.
        depth_tolerance = max(3.0 * robust_sigma, 0.10)

        # Remove background and foreground contamination.
        valid_mask &= (np.abs(depth_m - median_depth) <= depth_tolerance)
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count < 30: return None
        points_camera = self.deproject_depth_pixels(depth_m=depth_m, valid_mask=valid_mask, camera_info=camera_info)
        if points_camera is None: return None

        # Robust visible-surface center.
        center_camera = np.median(points_camera, axis=0)
        distances_from_center = np.linalg.norm(points_camera - center_camera, axis=1)

        # Use a percentile instead of the maximum so one bad
        # depth point cannot create an enormous object.
        radius_m = float(np.percentile(distances_from_center, 90.0))

        # Avoid zero-sized objects.
        radius_m = max(radius_m, 0.10)
        depth_uncertainty_m = max(robust_sigma, 0.05)

        return {
            "center_camera": center_camera,
            "radius_m": radius_m,
            "depth_uncertainty_m": float(depth_uncertainty_m),
            "valid_depth_pixels": valid_count
        }

    @staticmethod
    def depth_to_meters(depth_image, encoding):
        encoding = encoding.upper()

        if encoding in ("16UC1", "MONO16"):
            # Most RGB-D cameras use millimetres.
            return (depth_image.astype(np.float32) / 1000.0)

        if encoding == "32FC1":
            # Normally already in metres.
            return depth_image.astype(np.float32)

        raise ValueError(
            f"Unsupported depth encoding: {encoding}"
        )

    @staticmethod
    def extract_crop(image, xmin, ymin, xmax, ymax):
        height, width = image.shape[:2]

        x1 = max(0, int(xmin))
        y1 = max(0, int(ymin))
        x2 = min(width, int(xmax))
        y2 = min(height, int(ymax))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def extract_appearance_descriptor(crop):
        """Return a compact HSV descriptor used during reidentification."""
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        descriptor = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        descriptor = cv2.normalize(descriptor, None, alpha=1.0, norm_type=cv2.NORM_L1).reshape(-1)

        return descriptor.astype(np.float32)

    @staticmethod
    def appearance_distance(first_descriptor, second_descriptor):
        """Bhattacharyya distance: zero is identical, one is dissimilar."""
        if first_descriptor is None or second_descriptor is None:
            return 0.5

        first = np.asarray(first_descriptor, dtype=np.float32)
        second = np.asarray(second_descriptor, dtype=np.float32)

        return float(cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def elapsed_seconds(world_object, timestamp_ns):
        return (int(timestamp_ns) - int(world_object["last_seen_ns"])) / 1e9

    def predict_world_object_position(self, world_object, timestamp_ns):
        position = np.asarray(world_object["position_odom"], dtype=np.float64)
        velocity = np.asarray(world_object.get("velocity_odom", [0.0, 0.0, 0.0]), dtype=np.float64)
        prediction_time = min(max(0.0, self.elapsed_seconds(world_object, timestamp_ns)), self.maximum_prediction_time_s)

        return position + velocity * prediction_time

    def association_distance_limit(self, world_object, observation):
        elapsed = max(0.0, self.elapsed_seconds(world_object, observation["timestamp_ns"]))
        uncertainty = float(world_object["position_uncertainty_m"]) + float(observation["position_uncertainty_m"])
        geometry_margin = 0.25 * (float(world_object["radius_m"]) + float(observation["radius_m"]))

        if observation["label"] in self.dynamic_labels or world_object["label"] in self.dynamic_labels:
            return min(
                self.dynamic_max_association_distance_m,
                self.dynamic_base_association_distance_m
                + self.dynamic_max_speed_mps * elapsed
                + uncertainty
                + geometry_margin,
            )

        return self.static_association_distance_m + uncertainty + geometry_margin

    def association_cost(self, world_object_id, observation):
        world_object = self.world_objects[world_object_id]

        track_id = observation["track_id"]
        mapped_id = self.track_to_world.get(track_id) if track_id is not None else None
        labels_match = world_object["label"] == observation["label"]

        elapsed = self.elapsed_seconds(world_object, observation["timestamp_ns"])
        if elapsed < -0.05:
            return self.invalid_association_cost

        if (observation["label"] in self.dynamic_labels or world_object["label"] in self.dynamic_labels) and elapsed > self.dynamic_reidentification_age_s:
            return self.invalid_association_cost

        predicted_position = self.predict_world_object_position(world_object, observation["timestamp_ns"])
        observed_position = np.asarray(observation["position_odom"], dtype=np.float64)
        distance = float(np.linalg.norm(observed_position - predicted_position))
        distance_limit = self.association_distance_limit(world_object, observation)
        if distance > distance_limit:
            return self.invalid_association_cost

        spatial_cost = distance / max(distance_limit, 1e-6)
        visual_cost = self.appearance_distance(world_object.get("appearance_descriptor"), observation.get("appearance_descriptor"))

        # A detector label may flicker. Permit a label change only when the
        # tracker already maps the object, or geometry and appearance both
        # provide strong independent evidence.
        if not labels_match and mapped_id != world_object_id and not (spatial_cost <= self.label_change_max_spatial_cost and visual_cost <= self.label_change_max_appearance_distance):
            return self.invalid_association_cost

        cost = spatial_cost + self.appearance_cost_weight * visual_cost

        if not labels_match:
            cost += self.label_mismatch_cost

        if mapped_id == world_object_id:
            cost -= self.track_match_bonus

        return cost

    def create_world_object(self, observation):
        world_object_id = self.next_world_object_id
        self.next_world_object_id += 1
        track_id = observation["track_id"]
        vision_track_ids = set()

        if track_id is not None:
            vision_track_ids.add(track_id)
            self.track_to_world[track_id] = world_object_id
            self.track_last_seen_ns[track_id] = observation["timestamp_ns"]

        descriptor = observation.get("appearance_descriptor")
        self.world_objects[world_object_id] = {
            "world_object_id": world_object_id,
            "label": observation["label"],
            "label_scores": {observation["label"]: float(observation["confidence"])},
            "class_id": observation["class_id"],
            "label_confidence": observation["confidence"],
            "position_odom": observation["position_odom"].copy(),
            "velocity_odom": [0.0, 0.0, 0.0],
            "radius_m": observation["radius_m"],
            "position_uncertainty_m": observation["position_uncertainty_m"],
            "first_seen_ns": observation["timestamp_ns"],
            "last_seen_ns": observation["timestamp_ns"],
            "observation_count": 1,
            "vision_track_ids": vision_track_ids,
            "last_track_id": track_id,
            "last_bbox_xyxy": observation["bbox_xyxy"],
            "crop": observation["crop"],
            "appearance_descriptor": descriptor.tolist() if descriptor is not None else None,
            "active": True,
        }
        return world_object_id

    def update_matched_world_object(self, world_object_id, observation):
        world_object = self.world_objects[world_object_id]
        old_position = np.asarray(world_object["position_odom"], dtype=np.float64)
        new_position = np.asarray(observation["position_odom"], dtype=np.float64)
        elapsed = self.elapsed_seconds(world_object, observation["timestamp_ns"])

        if elapsed > 0.01:
            measured_velocity = (new_position - old_position) / elapsed
            measured_speed = float(np.linalg.norm(measured_velocity))
            if measured_speed <= self.maximum_velocity_update_mps:
                old_velocity = np.asarray(world_object.get("velocity_odom", [0.0, 0.0, 0.0]), dtype=np.float64)
                world_object["velocity_odom"] = (0.7 * old_velocity + 0.3 * measured_velocity).tolist()

        new_descriptor = observation.get("appearance_descriptor")
        old_descriptor = world_object.get("appearance_descriptor")
        if new_descriptor is not None:
            new_descriptor = np.asarray(new_descriptor, dtype=np.float32)
            if old_descriptor is None:
                blended_descriptor = new_descriptor
            else:
                blended_descriptor = 0.8 * np.asarray(old_descriptor, dtype=np.float32) + 0.2 * new_descriptor
            descriptor_sum = float(np.sum(blended_descriptor))
            if descriptor_sum > 0.0:
                blended_descriptor /= descriptor_sum
            world_object["appearance_descriptor"] = (
                blended_descriptor.tolist()
            )

        world_object["position_odom"] = new_position.tolist()
        world_object["radius_m"] = observation["radius_m"]
        world_object["position_uncertainty_m"] = observation["position_uncertainty_m"]
        label_scores = world_object.setdefault("label_scores", {})
        observed_label = observation["label"]
        label_scores[observed_label] = label_scores.get(observed_label, 0.0) + float(observation["confidence"])
        world_object["label"] = max(label_scores, key=label_scores.get)
        if world_object["label"] == observed_label:
            world_object["class_id"] = observation["class_id"]
        world_object["label_confidence"] = observation["confidence"]
        world_object["last_seen_ns"] = observation["timestamp_ns"]
        world_object["last_bbox_xyxy"] = observation["bbox_xyxy"]
        world_object["crop"] = observation["crop"]
        world_object["observation_count"] += 1
        world_object["active"] = True

        track_id = observation["track_id"]
        if track_id is not None:
            self.track_to_world[track_id] = world_object_id
            self.track_last_seen_ns[track_id] = observation["timestamp_ns"]
            world_object["vision_track_ids"].add(track_id)
            world_object["last_track_id"] = track_id

    def prune_stale_track_mappings(self, timestamp_ns):
        maximum_age_ns = int(self.dynamic_reidentification_age_s * 1e9)
        for track_id, last_seen_ns in list(self.track_last_seen_ns.items()):
            if int(timestamp_ns) - int(last_seen_ns) <= maximum_age_ns:
                continue
            self.track_last_seen_ns.pop(track_id, None)
            self.track_to_world.pop(track_id, None)

    def mark_stale_world_objects(self, timestamp_ns):
        for world_object in self.world_objects.values():
            age = max(0.0, self.elapsed_seconds(world_object, timestamp_ns))
            maximum_age = self.dynamic_active_age_s if world_object["label"] in self.dynamic_labels else self.static_active_age_s
            world_object["active"] = age <= maximum_age

    def update_world_objects_for_frame(self, observations, timestamp_ns):
        """
        Perform a global one-to-one Hungarian assignment for one frame.

        The caller must already hold world_objects_lock. Returned IDs have the
        same order as observations.
        """
        if not observations:
            self.prune_stale_track_mappings(timestamp_ns)
            self.mark_stale_world_objects(timestamp_ns)
            return []

        self.prune_stale_track_mappings(timestamp_ns)
        candidate_ids = list(self.world_objects.keys())
        assigned_ids = [None] * len(observations)

        if candidate_ids:
            cost_matrix = np.full((len(observations), len(candidate_ids)), self.invalid_association_cost, dtype=np.float64)

            for observation_index, observation in enumerate(observations):
                for candidate_index, world_object_id in enumerate(candidate_ids):
                    cost_matrix[observation_index, candidate_index] = self.association_cost(world_object_id, observation)

            rows, columns = linear_sum_assignment(cost_matrix)
            for row, column in zip(rows, columns):
                if cost_matrix[row, column] >= self.invalid_association_cost:
                    continue
                assigned_ids[row] = candidate_ids[column]

        for index, observation in enumerate(observations):
            world_object_id = assigned_ids[index]
            if world_object_id is None:
                world_object_id = self.create_world_object(observation)
                assigned_ids[index] = world_object_id
            else:
                self.update_matched_world_object(world_object_id, observation)

        self.mark_stale_world_objects(timestamp_ns)
        return assigned_ids

    def get_active_world_objects_snapshot(self, query_timestamp_ns):
        """Return a thread-safe, freshness-filtered snapshot for SST."""
        snapshot = {}
        with self.world_objects_lock:
            for world_object_id, world_object in self.world_objects.items():
                relative_age = self.elapsed_seconds(world_object, query_timestamp_ns)
                if relative_age < -self.sst_future_observation_tolerance_s:
                    continue
                age = max(0.0, relative_age)
                maximum_age = self.dynamic_active_age_s if world_object["label"] in self.dynamic_labels else self.static_active_age_s
                if age > maximum_age:
                    continue

                copied_object = world_object.copy()
                copied_object["position_odom"] = world_object["position_odom"].copy()
                copied_object["velocity_odom"] = world_object["velocity_odom"].copy()
                copied_object["vision_track_ids"] = set(world_object["vision_track_ids"])
                snapshot[world_object_id] = copied_object

        return snapshot

def main(args=None):
    rclpy.init(args=args)

    camera_sub = CameraSub()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(camera_sub)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    finally:
        executor.shutdown()
        camera_sub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
