import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class SceneClassifier:
    """
    Visual scene analysis component for Agent 2.

    Responsibilities:
    - Extract representative frames from a video.
    - Generate visual descriptions for each frame using BLIP.
    """

    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base",
        num_frames=8,
    ):
        self.num_frames = num_frames

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"SceneClassifier device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_name)

        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        print("BLIP visual model loaded successfully!")

    def extract_frames(self, video_path):
        """
        Extract representative frames evenly distributed across the video.

        Args:
            video_path (str): Path to the input video.

        Returns:
            list: List of tuples (frame_number, RGB frame).
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError(
                "Video contains no readable frames."
            )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            min(self.num_frames, total_frames),
            dtype=int,
        )

        frames = []

        for frame_number in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_number)
            )

            success, frame = cap.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                (int(frame_number), frame_rgb)
            )

        cap.release()

        if not frames:
            raise ValueError(
                "Could not extract any frames from the video."
            )

        return frames

    def generate_caption(self, frame):
        """
        Generate a caption for a single frame using BLIP.

        Args:
            frame: RGB NumPy array.

        Returns:
            str: Generated scene description.
        """

        image = Image.fromarray(frame)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip()

    def analyze_frames(self, frames):
        """
        Generate descriptions for all extracted frames.

        Args:
            frames (list): List of (frame_number, frame).

        Returns:
            list: Frame descriptions.
        """

        descriptions = []

        for frame_number, frame in frames:

            caption = self.generate_caption(frame)

            descriptions.append(
                {
                    "frame_number": frame_number,
                    "caption": caption,
                }
            )

            print(
                f"Frame {frame_number}: {caption}"
            )

        return descriptions

    def analyze_video(self, video_path):
        """
        Extract frames and generate descriptions for the video.

        Args:
            video_path (str): Path to video.

        Returns:
            list: Frame descriptions.
        """

        frames = self.extract_frames(video_path)

        descriptions = self.analyze_frames(frames)

        return descriptions