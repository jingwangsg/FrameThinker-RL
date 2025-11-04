# Code Style and Conventions

## Copyright Headers
All Python files in the `verl/` module include copyright headers:
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Naming Conventions
- **Functions**: snake_case (e.g., `get_custom_reward_fn`, `run_ppo`, `load_model_and_processor`)
- **Classes**: PascalCase (e.g., `TaskRunner`, `RayPPOTrainer`, `FSDPCheckpointManager`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CONFIG`, `MAX_ITERATIONS`)
- **Private functions**: Leading underscore not consistently used

## Type Hints
- Type hints are used moderately but not consistently throughout the codebase
- When present, they follow standard Python typing conventions
- Example from model_merger.py:
```python
def function(arg: str, flag: bool) -> Tuple[int, int]:
    ...
```

## Docstrings
- Docstrings are present but not comprehensive
- When used, they follow simple format without strict adherence to any particular style (not NumPy/Google style)
- Module-level docstrings sometimes present (e.g., in main_ppo.py: `"""Note that we don't combine the main with ray_trainer..."""`)

## Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- No strict line separation between groups
- Example from main_ppo.py:
```python
import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
```

## Code Organization
- Use of hydra for configuration management with decorators: `@hydra.main(config_path="config", config_name="ppo_trainer")`
- Separation of concerns: main entry points separated from runner logic
- Heavy use of remote Ray actors for distributed execution

## String Formatting
- Mix of f-strings and .format()
- Example: `f"using customized reward function '{function_name}' from '{file_path}'"`

## Error Handling
- Custom exceptions raised with informative messages
- Try-except blocks used where appropriate
- Example:
```python
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
```

## Testing
- Test files follow `test_*.py` naming convention
- Located in `tests/` directory with subdirectories matching source structure
- Use of PyTorch distributed testing utilities
- Example: `test_fsdp_ckpt.py` for FSDP checkpoint testing