import torch
import torch.nn.functional as F
from detectron2.modeling.roi_heads import FastRCNNOutputLayers

class CustomBoxPredictor(FastRCNNOutputLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_novel_classes_during_inference = False
        self.allow_novel_classes_during_training = False


    def predict_probs(self, predictions, proposals):
        if not self.allow_novel_classes_during_training:
            return super().predict_probs(predictions, proposals)

        scores, _ = predictions
        probs = F.softmax(scores, dim=-1)

        # During inference, we sometimes wants to predict only known classes; also we need to add a fake "background"
        # class that exists in the default R-CNN implementation
        if not self.training:

            # In this mode, we want to evaluate performance on labeled dataset with known classes only; therefore,
            # we have to keep predictions for knowns only
            if not self.allow_novel_classes_during_inference:
                # Keep probabilities for known (+ fake background) classes only
                probs = probs[:, :self.num_classes + 1]

            # In this mode, we want to evaluate performance on all classes; for that we also need to create a fake
            # logit for "background" class as `fast_rcnn_inference` ignores the last element
            else:
                # Append fake logits for a background class
                probs = torch.cat(
                    [probs, torch.zeros((probs.shape[0], 1), dtype=probs.dtype).to(probs.device)],
                    axis=1
                )

        num_inst_per_image = [len(p) for p in proposals]
        probs = probs.split(num_inst_per_image, dim=0)
        return probs