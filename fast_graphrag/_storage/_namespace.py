import os
import shutil
import time
from typing import Any, Callable, List, Optional

from fast_graphrag._utils import logger


class Workspace:
    @staticmethod
    def new(working_dir: str, checkpoint: Optional[int] = None, keep_n: int = 3) -> "Workspace":
        return Workspace(working_dir, checkpoint, keep_n)

    def __init__(self, working_dir: str, checkpoint: Optional[int] = None, keep_n: int = 3):
        self.working_dir: str = working_dir
        self.keep_n: int = keep_n
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.checkpoints = sorted(
            (int(x.name) for x in os.scandir(self.working_dir) if x.is_dir() and not x.name.startswith("0__err_")),
            reverse=True,
        )
        self.current_load_checkpoint: Optional[int] = checkpoint or (self.checkpoints[0] if self.checkpoints else None)
        self.save_checkpoint: Optional[int] = None
        self.failed_checkpoints: List[str] = []

    def __del__(self):
        for checkpoint in self.failed_checkpoints:
            old_path = os.path.join(self.working_dir, checkpoint)
            new_path = os.path.join(self.working_dir, f"0__err_{checkpoint}")
            os.rename(old_path, new_path)

        if self.keep_n > 0:
            checkpoints = sorted((x.name for x in os.scandir(self.working_dir) if x.is_dir()), reverse=True)
            for checkpoint in checkpoints[self.keep_n :]:
                shutil.rmtree(os.path.join(self.working_dir, str(checkpoint)))

    def make_for(self, namespace: str) -> "Namespace":
        return Namespace(self, namespace)

    def get_load_path(self) -> Optional[str]:
        if self.current_load_checkpoint is None:
            return None
        return os.path.join(self.working_dir, str(self.current_load_checkpoint))

    def get_save_path(self) -> str:
        if self.save_checkpoint is None:
            self.save_checkpoint = int(time.time())
            if not os.path.exists(os.path.join(self.working_dir, str(self.save_checkpoint))):
                os.makedirs(os.path.join(self.working_dir, str(self.save_checkpoint)))
        return os.path.join(self.working_dir, str(self.save_checkpoint))

    def _rollback(self) -> bool:
        if self.current_load_checkpoint is None:
            return False
        # List all directories in the working directory and select the one
        # with the smallest number greater then the current load checkpoint.
        try:
            self.current_load_checkpoint = next(x for x in self.checkpoints if x < self.current_load_checkpoint)
            logger.warning("Rolling back to checkpoint: %s", self.current_load_checkpoint)
        except (StopIteration, ValueError):
            self.current_load_checkpoint = None
            logger.warning("No checkpoints to rollback to. Last checkpoint tried: %s", self.current_load_checkpoint)

        return True

    async def with_checkpoints(self, fn: Callable[[], Any]) -> Any:
        while True:
            try:
                return await fn()
            except Exception as e:
                logger.warning("Error occurred loading checkpoint: %s", e)
                if self.current_load_checkpoint is not None:
                    self.failed_checkpoints.append(str(self.current_load_checkpoint))
                if self._rollback() is False:
                    break


class Namespace:
    def __init__(self, workspace: Workspace, namespace: Optional[str] = None):
        self.namespace = namespace
        self.workspace = workspace

    def get_load_path(self, resource_name: str) -> Optional[str]:
        assert self.namespace is not None, "Namespace must be set to get resource load path."
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        return os.path.join(load_path, f"{self.namespace}_{resource_name}")

    def get_save_path(self, resource_name: str) -> str:
        assert self.namespace is not None, "Namespace must be set to get resource save path."
        return os.path.join(self.workspace.get_save_path(), f"{self.namespace}_{resource_name}")
