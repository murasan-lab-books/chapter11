#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os
from typing import List, Tuple, Optional, Dict, Any

# Hailo SDK インポート
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                               InferVStreams, ConfigureParams,
                               InputVStreamParams, OutputVStreamParams,
                               FormatType)
    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False

# Picamera2 インポート
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

# ライブラリ公開API
__all__ = ['YOLODetector', 'CameraManager', 'draw_detections', 'COCO_CLASSES']

# COCOデータセットのクラス名（80クラス）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class YOLODetector:
    """
    YOLO物体検出器（Hailo-8L対応）

    Args:
        model_path: HEFモデルファイルのパス
        conf_threshold: 信頼度閾値（0.0-1.0）
        target_classes: 検出対象のクラス名リスト（Noneで全クラス）
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 target_classes: Optional[List[str]] = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.class_names = COCO_CLASSES.copy()

        # 検出対象クラスの設定
        self._target_class_ids: Optional[set] = None
        if target_classes is not None:
            self._set_target_classes(target_classes)

        # Hailo初期化
        self.device = None
        self.hef = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.input_name = None
        self.output_name = None
        self._initialize_hailo()

    def _set_target_classes(self, class_names: List[str]) -> None:
        """検出対象クラスを設定"""
        class_ids = set()
        invalid_classes = []

        for name in class_names:
            name_lower = name.lower().strip()
            try:
                class_id = next(
                    i for i, n in enumerate(self.class_names)
                    if n.lower() == name_lower
                )
                class_ids.add(class_id)
            except StopIteration:
                invalid_classes.append(name)

        if invalid_classes:
            raise ValueError(f"無効なクラス名: {invalid_classes}")

        self._target_class_ids = class_ids

    def _initialize_hailo(self) -> None:
        """Hailoデバイスとモデルの初期化"""
        if not _HAILO_AVAILABLE:
            raise RuntimeError(
                "HailoRT SDKが利用できません。"
                "確認: hailortcli fw-control identify"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")

        try:
            self.device = VDevice()
            self.hef = HEF(self.model_path)

            # 入出力情報の取得
            input_vstream_infos = self.hef.get_input_vstream_infos()
            output_vstream_infos = self.hef.get_output_vstream_infos()
            self.input_name = input_vstream_infos[0].name
            self.output_name = output_vstream_infos[0].name

            # ネットワーク設定
            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoStreamInterface.PCIe)
            network_groups = self.device.configure(self.hef, configure_params)
            self.network_group = network_groups[0]

            # VStreamsパラメータ
            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group, format_type=FormatType.UINT8)
            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32)

        except Exception as e:
            raise RuntimeError(
                f"Hailoデバイスの初期化に失敗: {e}\n"
                "確認: hailortcli fw-control identify"
            ) from e

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        画像から物体検出を実行

        Args:
            image: BGR形式の入力画像 (numpy.ndarray)

        Returns:
            list: 検出結果 [{'bbox', 'class_id', 'class_name', 'confidence'}, ...]
        """
        # 前処理
        resized = cv2.resize(image, (640, 640))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        preprocessed = np.expand_dims(rgb_image.astype(np.uint8), axis=0)

        # 推論実行
        with self.network_group.activate():
            with InferVStreams(self.network_group,
                              self.input_vstreams_params,
                              self.output_vstreams_params) as infer_pipeline:
                input_dict = {self.input_name: preprocessed}
                output_dict = infer_pipeline.infer(input_dict)
                outputs = output_dict[self.output_name]

        # 後処理
        return self._postprocess(outputs, image.shape[:2])

    def _postprocess(self, outputs: List,
                     original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """推論結果の後処理"""
        detections = []
        if not outputs or len(outputs) == 0:
            return detections

        batch_output = outputs[0]
        orig_h, orig_w = original_shape

        for class_id, class_detections in enumerate(batch_output):
            # クラスフィルタリング
            if self._target_class_ids is not None:
                if class_id not in self._target_class_ids:
                    continue

            if not isinstance(class_detections, np.ndarray) or class_detections.size == 0:
                continue

            for detection in class_detections:
                y1_norm, x1_norm, y2_norm, x2_norm, confidence = detection

                if confidence < self.conf_threshold:
                    continue

                # 座標変換
                x1 = max(0, min(int(x1_norm * orig_w), orig_w - 1))
                y1 = max(0, min(int(y1_norm * orig_h), orig_h - 1))
                x2 = max(0, min(int(x2_norm * orig_w), orig_w))
                y2 = max(0, min(int(y2_norm * orig_h), orig_h))

                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id]
                    })

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections


class CameraManager:
    """
    カメラ管理クラス（Camera Module V3用）

    Args:
        resolution: カメラ解像度 (width, height)
        device_id: カメラデバイスID
        flip_vertical: 上下反転するかどうか
    """

    def __init__(self, resolution: Tuple[int, int] = (1280, 720),
                 device_id: int = 0, flip_vertical: bool = False):
        self.resolution = resolution
        self.device_id = device_id
        self.flip_vertical = flip_vertical
        self.camera = None

        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "Picamera2が利用できません。"
                "確認: sudo apt install python3-picamera2"
            )

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Picamera2の初期化"""
        self.camera = Picamera2()
        camera_config = self.camera.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.camera.configure(camera_config)
        self.camera.start()

    def read_frame(self) -> Optional[np.ndarray]:
        """
        カメラからフレームを読み取る

        Returns:
            フレーム画像（BGRフォーマット）、失敗時はNone
        """
        if self.camera is None:
            return None

        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.flip_vertical:
            frame = cv2.flip(frame, 0)

        return frame

    def release(self) -> None:
        """カメラリソースの解放"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    画像に検出結果を描画

    Args:
        image: 入力画像（BGRフォーマット）
        detections: 検出結果のリスト
        color: バウンディングボックスの色（BGR）
        thickness: 線の太さ

    Returns:
        描画済み画像
    """
    result = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']}: {det['confidence']:.2f}"

        # バウンディングボックス
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # ラベル
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 > 20 else y1 + label_size[1] + 10

        cv2.rectangle(result, (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(result, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 + Hailo-8L YOLO物体検出"
    )
    parser.add_argument('--model', type=str, default='models/yolov8s_h8l.hef',
                       help='HEFモデルファイルのパス')
    parser.add_argument('--res', type=str, default='1280x720',
                       help='カメラ解像度 (例: 1280x720)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='信頼度閾値')
    parser.add_argument('--device', type=int, default=0,
                       help='カメラデバイスID')
    parser.add_argument('--flip', action='store_true',
                       help='カメラ映像を上下反転')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                       help='検出対象クラス (例: --classes person car)')
    args = parser.parse_args()

    # 解像度パース
    try:
        width, height = map(int, args.res.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"エラー: 無効な解像度フォーマット: {args.res}")
        return

    print("=== Raspberry Pi Hailo-8L YOLO Detector ===")
    print(f"モデル: {args.model}")
    print(f"解像度: {resolution[0]}x{resolution[1]}")
    print(f"信頼度閾値: {args.conf}")

    camera = None
    try:
        detector = YOLODetector(args.model, args.conf, target_classes=args.classes)
        camera = CameraManager(resolution, args.device, flip_vertical=args.flip)

        # FPS計算用
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0.0

        print("\n物体検出を開始します。'q'キーで終了。")

        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            # 推論
            inference_start = time.time()
            detections = detector.detect(frame)
            inference_time = (time.time() - inference_start) * 1000

            # 描画
            result = draw_detections(frame, detections)

            # FPS表示
            info = f"FPS: {current_fps:.1f} | Inference: {inference_time:.1f}ms"
            cv2.putText(result, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Hailo YOLO Detection', result)

            # FPS計算
            fps_counter += 1
            if fps_counter % 10 == 0:
                current_fps = 10 / (time.time() - fps_time)
                fps_time = time.time()

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    except FileNotFoundError as e:
        print(f"エラー: {e}")
    except RuntimeError as e:
        print(f"エラー: {e}")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("終了しました")


if __name__ == "__main__":
    main()
