import logging
import os

import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

from tracklab.datastruct import TrackerState
from tracklab.main import close_environment, init_environment
from tracklab.pipeline import Pipeline


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="live")
def main(cfg):
    device = init_environment(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    log.info(f"Live dataset initialized with splits: {list(tracking_dataset.sets.keys())}")

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    for module in modules:
        if module.training_enabled:
            module.train(tracking_dataset, pipeline, evaluator, OmegaConf.to_container(cfg.dataset, resolve=True))

    if cfg.test_tracking:
        log.info(f"Starting live tracking on source: {cfg.dataset.source}")
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        log.info(f"Live tracking split '{cfg.dataset.eval_set}' contains {len(tracking_set.video_metadatas)} video entries")
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
            source=cfg.dataset.source,
        )
        log.info(f"Live tracking engine instantiated: {type(tracking_engine).__name__}")
        tracking_engine.track_dataset()
        log.info("Live tracking finished")

    close_environment()
    return 0


if __name__ == "__main__":
    main()
