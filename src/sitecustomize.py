"""Installs STOPD's verl patches in every Python process that has this directory on PYTHONPATH.

verl forwards PYTHONPATH and the environment to its Ray workers, where rollouts
are generated and losses are computed, so patching only the driver would leave
the workers unchanged. Python imports sitecustomize at startup; rather than
importing verl there, an import hook applies each patch right after the verl
module it extends has finished loading, so processes that never import those
modules pay nothing. Nothing is installed unless STOPD_MODE is set.

If your environment already has a sitecustomize module, this one shadows it;
merge the two in that case.
"""

import os
import sys
from importlib.abc import MetaPathFinder


def _after_losses():
    import stopd.loss  # noqa: F401  (registers the stopd_k1 loss)


def _after_server():
    from stopd import verl_patch

    verl_patch.install_server()


def _after_trainer():
    from stopd import verl_patch

    verl_patch.install_trainer()


def _after_agent_loop():
    from stopd import verl_patch

    verl_patch.install_agent()


def _refuse_v1_trainer():
    # verl imports this package only for its V1 trainer, which never calls the patched _update_actor.
    raise RuntimeError("STOPD patches verl's legacy trainer; run with trainer.use_v1=False")


# verl module -> what to run once it has loaded
_TARGETS = {
    "verl.trainer.distillation.losses": _after_losses,
    "verl.workers.rollout.vllm_rollout.vllm_async_server": _after_server,
    "verl.trainer.ppo.ray_trainer": _after_trainer,
    "verl.experimental.agent_loop.agent_loop": _after_agent_loop,
    "verl.trainer.ppo.v1": _refuse_v1_trainer,
}


class _STOPDInstallHook(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        callback = _TARGETS.get(fullname)
        if callback is None:
            return None
        # Let the other finders locate the module, then run the patch right after its body executes.
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        real_exec_module = spec.loader.exec_module

        def exec_module(module):
            real_exec_module(module)
            callback()  # a patch that fails to install stops the run instead of training without it
            _TARGETS.pop(fullname, None)
            if not _TARGETS and _hook in sys.meta_path:
                sys.meta_path.remove(_hook)

        spec.loader.exec_module = exec_module
        return spec


_hook = _STOPDInstallHook()
if os.environ.get("STOPD_MODE"):
    sys.meta_path.insert(0, _hook)
