import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import hashlib
import argparse
import json
import pickle
import os
import time
from collections import Counter, deque


class MemeMatcher:
    # MediaPipe landmark indices
    LEFT_EYE_UPPER = [159, 145, 158]
    LEFT_EYE_LOWER = [23, 27, 133]
    RIGHT_EYE_UPPER = [386, 374, 385]
    RIGHT_EYE_LOWER = [253, 257, 362]
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
    MOUTH_INNER = [78, 308, 95, 88]
    NOSE_TIP = 4

    CACHE_FILE = "meme_features_cache.pkl"
    CACHE_VERSION = 5
    SAMPLES_FILE = "match_samples.jsonl"
    PROFILE_FILE = "match_profile.json"
    SPEED_SMILE_THRESHOLD = 0.28
    SPEED_BLINK_THRESHOLD = 0.35
    SPEED_EYE_OPENNESS_THRESHOLD = 0.40
    GESTURE_MEMES = {
        "cat pointing laugh": "point_and_laugh",
        "cat smile": "smile",
        "cover mouth": "cover_mouth",
        "speed my mom is kinda homeless face": "closed_eye_smile",
    }
    BASE_DIR = Path(__file__).resolve().parent

    def __init__(self, assets_folder="assets", frame_skip=2, meme_height=480):
        self.last_features = None
        self.last_face_landmarks = None
        self.last_face_blendshapes = None
        self.missing_face_frames = 0
        self.frame_counter = 0
        self.last_timestamp_ms = 0
        self.frame_skip = frame_skip
        self.meme_height = meme_height

        # Download or load models
        self.face_model_path = self._download_model(
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        self.hand_model_path = self._download_model(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

        # MediaPipe landmarkers
        self.face_mesh_video = self._init_face_landmarker(video_mode=True)
        self.hand_detector_video = self._init_hand_landmarker(video_mode=True)
        self.face_mesh_image = self._init_face_landmarker(video_mode=False)
        self.hand_detector_image = self._init_hand_landmarker(video_mode=False)

        # Meme storage
        self.memes = []
        self.meme_features = []
        self.gesture_memes = {}

        # Feature vectors for similarity computation
        self.feature_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'cheers_score',
            'hand_raised', 'num_hands', 'eye_openness', 'eyes_symmetry',
            'mouth_openness', 'mouth_width_ratio', 'mouth_elevation',
            'eyebrow_height', 'brow_symmetry'
        ]
        self.feature_weights = np.array([25, 20, 20, 30, 25, 15, 20, 10, 25, 20, 15, 20, 10])
        self.feature_factors = np.array([10, 10, 10, 10, 15, 15, 5, 5, 5, 5, 5, 5, 5])
        self.default_weights = self.feature_weights.copy()
        self.smile_threshold = 0.45
        self.minimum_score = 120.0
        self.minimum_margin = 5.0
        self.recent_matches = deque(maxlen=5)
        self.stable_match = None
        self.speed_hold_until = 0.0

        # Load memes (with caching)
        self.load_memes(assets_folder)
        self.load_profile()

    # ---------------- Model Initialization ----------------
    def _download_model(self, model_path, url):
        import subprocess
        model_path = str(self.BASE_DIR / model_path)
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print(f"{model_path} downloaded successfully!")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download model. Please download manually from {url}")
        return model_path

    def _init_face_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
                running_mode=mode,
                num_faces=1,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5 if video_mode else 0.3,
                min_face_presence_confidence=0.5 if video_mode else 0.3,
                min_tracking_confidence=0.5 if video_mode else 0.0
            )
        )

    def _init_hand_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3 if video_mode else 0.0
            )
        )

    # ---------------- Meme Loading with Cache ----------------
    def load_memes(self, folder):
        assets_path = Path(folder)
        if not assets_path.is_absolute():
            assets_path = self.BASE_DIR / assets_path
        assets_path = assets_path.resolve()
        if not assets_path.is_dir():
            raise FileNotFoundError(f"Meme assets folder not found: {assets_path}")
        image_files = sorted(
            p for p in assets_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )
        signature = {
            "version": self.CACHE_VERSION,
            "folder": str(assets_path),
            "height": self.meme_height,
            "images": [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in image_files],
        }
        cache_path = self.BASE_DIR / self.CACHE_FILE
        self.memes, self.meme_features, self.gesture_memes = [], [], {}
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and cached.get("signature") == signature:
                    self.memes, self.meme_features = cached["memes"], cached["features"]
                    self.gesture_memes = cached["gesture_memes"]
                    print(f"Loaded {len(self.memes) + len(self.gesture_memes)} memes from cache.\n")
                    return
                print("Meme images or settings changed. Rebuilding cache...")
            except (OSError, EOFError, pickle.UnpicklingError, KeyError, ValueError):
                print("Could not read meme cache. Rebuilding...")
        print(f"Found {len(image_files)} meme images. Extracting features...")

        def process_meme(img_file):
            img = cv2.imread(str(img_file))
            if img is None:
                return None
            h, w = img.shape[:2]
            scale = self.meme_height / h
            img_resized = cv2.resize(img, (int(w * scale), self.meme_height))
            meme = {'image': img_resized, 'name': img_file.stem.replace('_', ' ').title(),
                    'path': str(img_file)}
            gesture = self.GESTURE_MEMES.get(img_file.stem.replace('_', ' ').casefold())
            if gesture:
                return meme, gesture
            features = self.extract_face_features(img_resized, is_static=True)
            if features is None:
                print(f"Skipping {img_file.name}: No face detected")
                return None
            return meme, features

        # Reuse the image landmarkers sequentially instead of sharing them across threads.
        for img_file in image_files:
            r = process_meme(img_file)
            if r:
                meme, features = r
                if isinstance(features, str):
                    self.gesture_memes[features] = meme
                else:
                    self.memes.append(meme)
                    self.meme_features.append(features)
                print(f"Loaded: {meme['name']}")

        with cache_path.open("wb") as f:
            pickle.dump({"signature": signature, "memes": self.memes,
                         "features": self.meme_features,
                         "gesture_memes": self.gesture_memes}, f)
        print(f"Total memes loaded: {len(self.memes) + len(self.gesture_memes)}\n")
        if not self.memes and not self.gesture_memes:
            print("No usable meme images found.")

    def meme_names(self):
        return [m["name"] for m in self.memes] + [m["name"] for m in self.gesture_memes.values()]

    def load_profile(self):
        self.prototypes = {}
        profile_path = self.BASE_DIR / self.PROFILE_FILE
        if not profile_path.exists():
            return
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if (profile["version"] != self.CACHE_VERSION or
                    profile["memes"] != sorted(self.meme_names())):
                print("Matching profile is out of date; use --calibrate to rebuild it.")
                return
            weights = np.asarray(profile["weights"], dtype=float)
            if weights.shape != self.default_weights.shape or not np.all(np.isfinite(weights)):
                raise ValueError("Invalid profile weights")
            self.feature_weights = weights
            self.prototypes = profile["prototypes"]
            self.smile_threshold = float(profile["smile_threshold"])
            self.minimum_score = float(profile["minimum_score"])
            print("Loaded matching calibration.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Could not use matching profile: {exc}")

    def read_samples(self):
        path = self.BASE_DIR / self.SAMPLES_FILE
        if not path.exists():
            return []
        allowed = set(self.meme_names()) | {"neutral"}
        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if (row["version"] == self.CACHE_VERSION and row["label"] in allowed
                        and all(k in row["features"] for k in self.feature_keys)):
                    samples.append(row)
            except (ValueError, KeyError, TypeError):
                continue
        return samples

    def collect_samples(self, label):
        if label not in set(self.meme_names()) | {"neutral"}:
            raise ValueError(f"Unknown label: {label}. Choose from {self.meme_names()} or neutral")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        print(f"Collecting {label}. Make the expression, press SPACE to save; Q to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                features = self.extract_face_features(frame)
                display = frame.copy()
                cv2.putText(display, f"{label}: SPACE save / Q quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if features is None:
                    cv2.putText(display, "No face detected", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow("Collect matching examples", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 32 and features is not None:
                    row = {"version": self.CACHE_VERSION, "label": label,
                           "features": {k: float(v) for k, v in features.items()}}
                    with (self.BASE_DIR / self.SAMPLES_FILE).open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"Saved {label} example ({len(self.read_samples())} total).")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def calibrate(self):
        samples = self.read_samples()
        counts = Counter(row["label"] for row in samples)
        if not any(counts[name] >= 3 for name in self.meme_names()):
            raise ValueError("Collect at least 3 examples for a meme before calibrating.")
        human_names = {m["name"] for m in self.memes}
        groups = {name: [row["features"] for row in samples if row["label"] == name]
                  for name in human_names}
        self.prototypes = {}
        for name, rows in groups.items():
            if len(rows) >= 3:
                self.prototypes[name] = {
                    key: float(np.median([row[key] for row in rows]))
                    for key in self.feature_keys
                }

        trained = [rows for rows in groups.values() if len(rows) >= 3]
        if len(trained) >= 2:
            centers = np.array([[np.median([row[k] for row in rows])
                                 for k in self.feature_keys] for rows in trained])
            residuals = np.array([
                [row[k] - centers[i, j] for j, k in enumerate(self.feature_keys)]
                for i, rows in enumerate(trained) for row in rows
            ])
            variance_ratio = np.var(centers, axis=0) / (np.var(residuals, axis=0) + 1e-4)
            scale = np.clip(variance_ratio / (np.median(variance_ratio) + 1e-6), 0.5, 2.0)
            self.feature_weights = self.default_weights * scale
            self.feature_weights *= self.default_weights.sum() / self.feature_weights.sum()

        def best_threshold(positive, negative, values, default):
            if len(positive) < 3 or len(negative) < 3:
                return default
            candidates = [default] + list(values)
            return max(candidates, key=lambda t: (
                sum(v >= t for v in positive) / len(positive) +
                sum(v < t for v in negative) / len(negative)))

        smile_name = self.gesture_memes.get("smile", {}).get("name")
        smile_positive = [row["features"]["smiling"] for row in samples
                          if row["label"] == smile_name and row["features"]["laughing"] < 0.35
                          and not self.is_speed_expression(row["features"])]
        smile_negative = [row["features"]["smiling"] for row in samples
                          if row["label"] != smile_name and row["features"]["laughing"] < 0.35
                          and not self.is_speed_expression(row["features"])]
        self.smile_threshold = best_threshold(
            smile_positive, smile_negative, np.linspace(0.15, 0.7, 56), 0.45)

        # Neutral examples teach the app when the closest human meme is still a poor match.
        positives = [self.best_human_score(row["features"]) for row in samples
                     if row["label"] in self.prototypes]
        neutrals = [self.best_human_score(row["features"]) for row in samples
                    if row["label"] == "neutral"]
        if len(positives) >= 3 and len(neutrals) >= 3:
            thresholds = np.linspace(min(positives + neutrals), max(positives + neutrals), 100)
            self.minimum_score = float(max(thresholds, key=lambda t: (
                sum(v >= t for v in positives) / len(positives) +
                sum(v < t for v in neutrals) / len(neutrals))))

        profile = {"version": self.CACHE_VERSION, "memes": sorted(self.meme_names()),
                   "weights": self.feature_weights.tolist(), "prototypes": self.prototypes,
                   "smile_threshold": float(self.smile_threshold),
                   "minimum_score": self.minimum_score}
        (self.BASE_DIR / self.PROFILE_FILE).write_text(
            json.dumps(profile, indent=2), encoding="utf-8")
        print("Calibration saved. Example counts:", dict(sorted(counts.items())))
        print(f"Cat smile threshold: {self.smile_threshold:.2f}; "
              f"human minimum score: {self.minimum_score:.1f}")
        print("Collect at least 3 examples per meme and 3 neutral examples for full calibration.")
        self.evaluate_samples(samples)

    def evaluate_samples(self, samples=None):
        samples = self.read_samples() if samples is None else samples
        if not samples:
            print("No examples to evaluate.")
            return
        correct = 0
        mistakes = Counter()
        for row in samples:
            match, _ = self.find_best_match(row["features"])
            predicted = match["name"] if match else "neutral"
            if predicted == row["label"]:
                correct += 1
            else:
                mistakes[(row["label"], predicted)] += 1
        print(f"Collected-example accuracy: {correct}/{len(samples)} "
              "(these examples were also used for calibration).")
        for (expected, predicted), count in mistakes.most_common():
            print(f"  {expected} predicted as {predicted}: {count}")

    # ---------------- Feature Extraction ----------------
    def extract_face_features(self, image, is_static=False):
        if is_static:
            face_landmarker = self.face_mesh_image
            hand_landmarker = self.hand_detector_image
        else:
            face_landmarker = self.face_mesh_video
            hand_landmarker = self.hand_detector_video

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if is_static:
            face_res = face_landmarker.detect(mp_image)
            hand_res = hand_landmarker.detect(mp_image)
        else:
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return getattr(self, "last_features", None)
            timestamp_ms = max(time.monotonic_ns() // 1_000_000,
                               self.last_timestamp_ms + 1)
            self.last_timestamp_ms = timestamp_ms
            face_res = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width = image.shape[:2]
        if not face_res.face_landmarks:
            if not is_static:
                self.missing_face_frames += 1
                # A hand over the mouth can briefly hide the face from MediaPipe.
                if (self.missing_face_frames <= 3 and self.last_face_landmarks is not None
                        and hand_res.hand_landmarks):
                    covered = self._compute_features(self.last_face_landmarks, hand_res,
                                                     self.last_face_blendshapes,
                                                     (width, height))
                    if covered['covering_mouth']:
                        self.last_features = covered
                        return covered
                self.last_features = None
            return None

        landmarks = face_res.face_landmarks[0]
        # MediaPipe coordinates are fractions of the image dimensions. Convert
        # both axes to pixels before measuring distances across them.
        landmark_array = np.array([[l.x * width, l.y * height] for l in landmarks])
        blendshapes = {category.category_name: category.score
                       for category in face_res.face_blendshapes[0]}
        features = self._compute_features(landmark_array, hand_res, blendshapes,
                                          (width, height))
        if not is_static:
            self.last_face_landmarks = landmark_array
            self.last_face_blendshapes = blendshapes
            self.missing_face_frames = 0
            self.last_features = features
        return features

    def _compute_features(self, landmark_array, hand_res, blendshapes, image_size):
        width, height = image_size
        # Eye separation scales with the face, unlike the image dimensions.
        eye_spacing = np.linalg.norm(landmark_array[33] - landmark_array[263]) + 1e-6

        # Eye aspect ratios
        def ear(upper, lower, corners):
            vert = np.linalg.norm(landmark_array[upper] - landmark_array[lower], axis=1).mean()
            horiz = np.linalg.norm(landmark_array[corners[0]] - landmark_array[corners[1]])
            return vert / (horiz + 1e-6)

        left_ear = ear(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER, (33, 133))
        right_ear = ear(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER, (362, 263))
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth
        mouth_top, mouth_bottom = landmark_array[13], landmark_array[14]
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_left, mouth_right = landmark_array[61], landmark_array[291]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ar = mouth_height / (mouth_width + 1e-6)
        inner_width = np.linalg.norm(landmark_array[78] - landmark_array[308])
        mouth_width_ratio = inner_width / (mouth_width + 1e-6)

        # Eyebrows
        left_brow_y = landmark_array[self.LEFT_EYEBROW][:, 1].mean()
        right_brow_y = landmark_array[self.RIGHT_EYEBROW][:, 1].mean()
        left_eye_center = landmark_array[self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER][:, 1].mean()
        right_eye_center = landmark_array[self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER][:, 1].mean()
        left_brow_h = (left_eye_center - left_brow_y) / eye_spacing
        right_brow_h = (right_eye_center - right_brow_y) / eye_spacing
        avg_brow_h = (left_brow_h + right_brow_h) / 2.0

        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
        nose_tip = landmark_array[self.NOSE_TIP]
        mouth_elev = (nose_tip[1] - mouth_center_y) / eye_spacing

        # Hands
        num_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
        hand_raised = 0.0
        pointing = 0.0
        covering_mouth = 0.0
        if num_hands > 0:
            face_center = landmark_array[:, 1].mean()
            face_top = landmark_array[:, 1].min()
            face_height = landmark_array[:, 1].max() - face_top
            wrist_y = np.array([h[0].y * height for h in hand_res.hand_landmarks])
            middle_y = np.array([h[12].y * height for h in hand_res.hand_landmarks])
            if np.any((middle_y < face_center + 0.2 * face_height) |
                      (wrist_y < face_top + 0.3 * face_height)):
                hand_raised = 1.0
            for hand in hand_res.hand_landmarks:
                mouth_center = (mouth_top + mouth_bottom) / 2.0
                near_mouth = sum(
                    np.linalg.norm(np.array([hand[i].x * width, hand[i].y * height])
                                   - mouth_center) < 0.7 * eye_spacing
                    for i in (5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20)
                )
                if near_mouth >= 3:
                    covering_mouth = 1.0

                def finger_extended(tip, pip, mcp):
                    def distance(a, b):
                        return np.linalg.norm(np.array([
                            (hand[a].x - hand[b].x) * width,
                            (hand[a].y - hand[b].y) * height,
                            (hand[a].z - hand[b].z) * width,
                        ]))
                    return distance(tip, mcp) > 1.15 * distance(pip, mcp)

                folded = sum(not finger_extended(*finger) for finger in
                             ((12, 10, 9), (16, 14, 13), (20, 18, 17)))
                palm_size = np.hypot((hand[9].x - hand[0].x) * width,
                                     (hand[9].y - hand[0].y) * height)
                finger_toward_camera = ((hand[6].z - hand[8].z) * width >
                                        0.3 * palm_size)
                if (finger_extended(8, 6, 5) or finger_toward_camera) and folded >= 2:
                    pointing = 1.0

        # Features
        return {
            'eye_openness': avg_ear,
            'left_eye_open': left_ear,
            'right_eye_open': right_ear,
            'eyes_symmetry': abs(left_ear - right_ear),
            'mouth_openness': mouth_ar,
            'mouth_width': mouth_width,
            'mouth_width_ratio': mouth_width_ratio,
            'mouth_elevation': mouth_elev,
            'eyebrow_height': avg_brow_h,
            'left_brow_height': left_brow_h,
            'right_brow_height': right_brow_h,
            'brow_symmetry': abs(left_brow_h - right_brow_h),
            'num_hands': num_hands,
            'hand_raised': hand_raised,
            'pointing': pointing,
            'covering_mouth': covering_mouth,
            'laughing': blendshapes.get('jawOpen', 0.0),
            'smiling': (blendshapes.get('mouthSmileLeft', 0.0) +
                        blendshapes.get('mouthSmileRight', 0.0)) / 2.0,
            'eyes_closed': (blendshapes.get('eyeBlinkLeft', 0.0) +
                            blendshapes.get('eyeBlinkRight', 0.0)) / 2.0,
            'surprise_score': avg_ear * avg_brow_h * mouth_ar,
            'smile_score': mouth_width_ratio * (1.0 - mouth_ar),
            'concern_score': avg_brow_h * (1.0 - mouth_elev),
            'cheers_score': mouth_width_ratio * (1.0 - mouth_ar) * hand_raised
        }

    # ---------------- Vectorized Similarity ----------------
    def compute_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0
        vec1 = np.array([features1[k] for k in self.feature_keys])
        vec2 = np.array([features2[k] for k in self.feature_keys])
        diff = np.abs(vec1 - vec2)
        similarity = np.exp(-diff * self.feature_factors)
        return float(np.sum(self.feature_weights * similarity))

    def human_scores(self, user_features):
        return np.array([
            self.compute_similarity(user_features,
                                    self.prototypes.get(meme["name"], features))
            for meme, features in zip(self.memes, self.meme_features)
        ])

    def best_human_score(self, user_features):
        scores = self.human_scores(user_features)
        return float(np.max(scores)) if len(scores) else 0.0

    def is_speed_expression(self, features):
        return (features.get('smiling', 0) >= self.SPEED_SMILE_THRESHOLD and
                (features.get('eyes_closed', 0) >= self.SPEED_BLINK_THRESHOLD or
                 features.get('eye_openness', 1) <= self.SPEED_EYE_OPENNESS_THRESHOLD))

    def find_best_match(self, user_features):
        if user_features is None:
            return None, 0.0
        if (user_features.get('pointing', 0) >= 0.5 and
                'point_and_laugh' in self.gesture_memes):
            return self.gesture_memes['point_and_laugh'], 255.0
        if (user_features.get('covering_mouth', 0) >= 0.5 and
                'cover_mouth' in self.gesture_memes):
            return self.gesture_memes['cover_mouth'], 255.0
        if (self.is_speed_expression(user_features) and
                'closed_eye_smile' in self.gesture_memes):
            return self.gesture_memes['closed_eye_smile'], 255.0
        if (user_features.get('smiling', 0) >= self.smile_threshold and
                user_features.get('laughing', 0) < 0.35 and
                not self.is_speed_expression(user_features) and
                'smile' in self.gesture_memes):
            return self.gesture_memes['smile'], 255.0
        scores = self.human_scores(user_features)
        if len(scores) == 0:
            return None, 0.0
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        runner_up = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
        if best_score < self.minimum_score or best_score - runner_up < self.minimum_margin:
            return None, best_score
        return self.memes[best_idx], best_score

    def stabilize_match(self, candidate):
        name = candidate["name"] if candidate else None
        speed = self.gesture_memes.get("closed_eye_smile")
        if candidate is not None and candidate is speed:
            self.stable_match = name
            self.speed_hold_until = time.monotonic() + 0.5
            self.recent_matches.clear()
            return candidate
        if speed is not None and time.monotonic() < self.speed_hold_until:
            return speed
        self.recent_matches.append(name)
        most_common, count = Counter(self.recent_matches).most_common(1)[0]
        if count >= 3 and most_common != self.stable_match:
            self.stable_match = most_common
        if self.stable_match is None:
            return None
        return next((m for m in self.memes + list(self.gesture_memes.values())
                     if m["name"] == self.stable_match), None)

    # ---------------- Main Loop ----------------
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n🎥 Camera started! Press 'q' to quit, 'd' for feature values\n")
        show_debug = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            user_features = self.extract_face_features(frame)
            candidate, score = self.find_best_match(user_features)
            best_meme = self.stabilize_match(candidate) if user_features else None
            if user_features is None:
                self.recent_matches.clear()
                self.stable_match = None
                self.speed_hold_until = 0.0
            elif best_meme and best_meme is not candidate:
                if any(m is best_meme for m in self.memes):
                    index = next(i for i, m in enumerate(self.memes) if m is best_meme)
                    score = float(self.human_scores(user_features)[index])
                else:
                    score = 255.0

            h, w = frame.shape[:2]
            if best_meme:
                meme_img = best_meme['image']
                meme_h, meme_w = meme_img.shape[:2]
                scale = h / meme_h
                new_w = int(meme_w * scale)
                meme_resized = cv2.resize(meme_img, (new_w, h))

                display = np.zeros((h, w + new_w, 3), dtype=np.uint8)
                display[:, :w] = frame
                display[:, w:w + new_w] = meme_resized

                # Overlay text
                cv2.rectangle(display, (5, 5), (200, 45), (0, 0, 0), -1)
                cv2.putText(display, "YOU", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.rectangle(display, (w + 5, 5), (w + new_w - 5, 75), (0, 0, 0), -1)
                cv2.putText(display, best_meme['name'], (w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                label = ("Gesture match" if any(m is best_meme for m in self.gesture_memes.values())
                         else f"Similarity: {score:.1f}")
                cv2.putText(display, label, (w + 10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)
            else:
                display = frame
                message = "No face detected" if user_features is None else "No close match"
                cv2.putText(display, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            if show_debug and user_features:
                detail = (f"Smile {user_features['smiling']:.2f}  "
                          f"Blink {user_features['eyes_closed']:.2f}  "
                          f"Eye {user_features['eye_openness']:.2f}")
                cv2.rectangle(display, (5, h - 36), (min(w - 5, 480), h - 2),
                              (0, 0, 0), -1)
                cv2.putText(display, detail, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)

            cv2.imshow("Meme Matcher - Press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                show_debug = not show_debug
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match webcam expressions to memes")
    parser.add_argument("--collect", metavar="LABEL", help="record labeled webcam examples")
    parser.add_argument("--calibrate", action="store_true", help="tune matching from examples")
    parser.add_argument("--evaluate", action="store_true", help="check collected examples")
    args = parser.parse_args()
    print("Meme Matcher Starting...\n")
    matcher = MemeMatcher(assets_folder="assets")
    if args.collect:
        matcher.collect_samples(args.collect)
    elif args.calibrate:
        matcher.calibrate()
    elif args.evaluate:
        matcher.evaluate_samples()
    else:
        matcher.run()
