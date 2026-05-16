from PIL import Image
import io
import numpy as np
import torch


from ultralytics import YOLO

from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


class Ensemble:
    def __init__(self):
        #self.detr = RFDETRMedium()
        self.yolo = YOLO("yolo26m.pt")
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.frcnn = fasterrcnn_resnet50_fpn(weights=weights).eval().cuda()

        #self.detr_name_to_id = {v: k for k, v in COCO_CLASSES.items()}
        self.yolo_name_to_id = {v: k for k, v in self.yolo.names.items()}

    # def call_detr(self, image, target_class_id) -> float:
    #     detections = self.detr.predict(image, threshold=0.1)

    #     # Mask detections matching the target class
    #     mask = detections.class_id == target_class_id
    #     matching_confidences = detections.confidence[mask]

    #     if len(matching_confidences) == 0:
    #         return 0.0
    #     return float(matching_confidences.max())

    def call_yolo(self, image, target_class_id) -> float:
        result = self.yolo(image, conf=0.0, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return 0.0

        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        matching = conf[cls == target_class_id]
        if matching.size == 0:
            return 0.0
        return float(matching.max())

    def call_resnet(self, image, target_class_id) -> float:
        x = to_tensor(image.convert("RGB")).cuda()
        with torch.no_grad():
            pred = self.frcnn([x])[0]
        mask = pred["labels"] == target_class_id
        if mask.sum() == 0:
            return 0.0
        return float(pred["scores"][mask].max().item())

    def __call__(self, image, target_class) -> float:
        """
        Usage:

        ensemble = Ensemble()

        images = ....
        do something so that `images` exists
        .
        .
        .
        for image in images:
            reward = ensemble(image, target_class)
        """
        #detr_id = self.detr_name_to_id[target_class]
        detr_id = 88
        yolo_id = self.yolo_name_to_id[target_class]

        #detr_score = self.call_detr(image, detr_id)
        yolo_score = self.call_yolo(image, yolo_id)
        resnet_score = self.call_resnet(image, detr_id)

        return -np.mean([yolo_score, resnet_score]) + 0.2


def ensemble_detector_score(unsafe_concept: str):
    """Reward factory that penalizes the presence of ``unsafe_concept`` (a COCO class name)
    in generated images. Higher detector confidence => lower (more negative) reward, so
    PPO is pushed *away* from generating the concept.
    """
    ensemble = Ensemble()

    def _fn(images, prompts, metadata):
        # DDPO passes a float tensor in [0,1] NCHW. The ensemble members expect PIL
        # (call_resnet uses image.convert("RGB")), so normalize once here.
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(img) for img in images]

        rewards = [float(ensemble(img, unsafe_concept)) for img in images]
        return np.array(rewards, dtype=np.float32), {}

    return _fn
