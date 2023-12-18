import torch
import logging
import datetime
import time

from tqdm import tqdm

from ground_dino_utils import inference_gdino

@torch.no_grad()
def inference(data_loader, evaluator, model, text_prompt, param_dict):
    evaluator.reset()
    _run_generic_evaluation_loop(data_loader, evaluator, model, text_prompt, param_dict)
    
    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results

def _run_generic_evaluation_loop(data_loader, evaluator, model, text_prompt, param_dict):
    logger = logging.getLogger(__name__)
    logger.info("###\n### Start inference on {} batches\n###".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)

    start_time = time.perf_counter()
    for idx, inputs in enumerate(tqdm(data_loader)):
        if idx == num_warmup:
            start_time = time.perf_counter()

        outputs = inference_gdino(model, inputs, text_prompt, param_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        evaluator.process(inputs, outputs)

        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
        if idx >= num_warmup * 2:
            eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            # log_every_n_seconds(
            #     logging.INFO,
            #     (
            #         f"Inference done {idx + 1}/{total}. "
            #         f"Total: {total_seconds_per_iter:.4f} s/iter. "
            #         f"ETA={eta}"
            #     ),
            #     n=5,
            # )
        
        # if idx != 0 and idx % 5000 == 0:
        #     torch.save(evaluator.evaluator._predictions, param_dict["out_dir"] + "/predictions_" + str(idx) + ".pt")

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info("Total inference time: {} ({:.6f} s / iter per device)".format(
        total_time_str, total_time / (total - num_warmup)
    ))