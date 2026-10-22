import json
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from .scene_classifier import SceneClassifier


class ContextAnalyzerAgent:
    """
    Agent 2: Context Analyzer.

    Pipeline:
        Video
          ↓
        Frame Sampling
          ↓
        BLIP Visual Descriptions
          ↓
        Qwen Scene Aggregation
          ↓
        Structured Scene Context
    """

    def __init__(
        self,
        num_frames=8,
        aggregator_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    ):

        self.scene_classifier = SceneClassifier(
            num_frames=num_frames
        )

        self.aggregator_model_name = aggregator_model_name

        print(
            f"Loading aggregator model: "
            f"{aggregator_model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            aggregator_model_name
        )

        self.aggregator_model = (
            AutoModelForCausalLM.from_pretrained(
                aggregator_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        )

        self.aggregator_model.eval()

        print(
            "Qwen aggregator model loaded successfully!"
        )

    def build_prompt(self, descriptions):
        """
        Build the prompt used by Qwen to aggregate
        frame-level descriptions.
        """

        all_descriptions = "\n".join(
            f"Frame {item['frame_number']}: "
            f"{item['caption']}"
            for item in descriptions
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scene understanding agent. "
                    "Analyze descriptions from multiple "
                    "video frames and return ONLY valid JSON. "
                    "Infer values only from the descriptions."
                ),
            },
            {
                "role": "user",
                "content": f"""
Analyze these frame descriptions:

{all_descriptions}

Return exactly this JSON structure:

{{
    "Scene Type": "actual scene type",
    "Main Objects": ["actual objects"],
    "Persistent Objects": ["objects appearing repeatedly"],
    "Visible Surfaces": ["actual surfaces"],
    "Spatial Relationships": ["actual relationships"],
    "Scene Context": "actual concise scene interpretation"
}}

Rules:

- Return ONLY valid JSON.
- Do not return explanations.
- Do not copy placeholders.
- Use only information supported by the frame descriptions.
- Do not invent objects or relationships.
""",
            },
        ]

        return messages

    def aggregate_descriptions(self, descriptions):
        """
        Use Qwen to combine frame-level descriptions
        into a structured scene representation.
        """

        messages = self.build_prompt(descriptions)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.aggregator_model.device)

        with torch.no_grad():

            generated_ids = (
                self.aggregator_model.generate(
                    **model_inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    use_cache=True,
                )
            )

        generated_ids = generated_ids[
            :,
            model_inputs.input_ids.shape[1]:
        ]

        output = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return self.parse_json_output(output)

    def parse_json_output(self, output):
        """
        Extract and parse JSON from the model output.
        """

        output = output.strip()

        # First try direct JSON parsing.
        try:
            return json.loads(output)

        except json.JSONDecodeError:
            pass

        # Try extracting a JSON object from extra text.
        match = re.search(
            r"\{.*\}",
            output,
            re.DOTALL
        )

        if match:

            json_text = match.group(0)

            try:
                return json.loads(json_text)

            except json.JSONDecodeError:
                pass

        # Return raw output if JSON parsing fails.
        return {
            "raw_output": output,
            "parse_error": True,
        }

    def analyze_video(self, video_path):
        """
        Run the complete Agent 2 pipeline.

        Args:
            video_path (str): Path to the video.

        Returns:
            dict: Structured scene analysis.
        """

        print("\n=== Agent 2: Context Analysis ===")

        print("\n[1/2] Extracting frames and generating captions...")

        descriptions = self.scene_classifier.analyze_video(
            video_path
        )

        print("\n[2/2] Aggregating scene information...")

        scene_context = self.aggregate_descriptions(
            descriptions
        )

        result = {
            "frame_descriptions": descriptions,
            "scene_analysis": scene_context,
        }

        print("\n=== Agent 2 completed successfully ===")

        return result


# Simple standalone test
if __name__ == "__main__":

    VIDEO_PATH = "/content/video3.mp4"

    agent = ContextAnalyzerAgent(
        num_frames=8
    )

    result = agent.analyze_video(
        VIDEO_PATH
    )

    print("\nFINAL OUTPUT:")
    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )