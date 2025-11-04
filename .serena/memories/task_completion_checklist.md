# Task Completion Checklist

When completing a coding task for FrameThinker-RL, follow these steps:

## 1. Code Quality
- [ ] Ensure copyright header is present in any new files in `verl/` module
- [ ] Follow naming conventions (snake_case for functions, PascalCase for classes)
- [ ] Add appropriate error handling with informative messages
- [ ] Use type hints where appropriate (though not strictly required)
- [ ] Keep imports organized (stdlib, third-party, local)

## 2. Testing (if applicable)
- [ ] Write tests for new functionality in appropriate `tests/` subdirectory
- [ ] Follow `test_*.py` naming convention
- [ ] Run relevant tests: `pytest tests/path/to/test_file.py`
- [ ] Ensure tests pass before committing

## 3. Configuration (if applicable)
- [ ] Update Hydra configs if adding new configurable parameters
- [ ] Document any new config parameters in comments or docstrings
- [ ] Test config overrides work correctly

## 4. Documentation
- [ ] Update relevant comments in code
- [ ] If adding new major functionality, consider if README.md needs updating
- [ ] Document any new environment variables or setup requirements

## 5. Git & Version Control
- [ ] Check git status: `git status`
- [ ] Review changes: `git diff`
- [ ] Stage changes: `git add <files>`
- [ ] Commit with descriptive message: `git commit -m "message"`
- [ ] Follow existing commit message style (lowercase, imperative mood)

## 6. Pre-commit Checks (if available)
- [ ] If pre-commit hooks are configured, ensure they pass
- [ ] Note: requirements.txt includes `pre-commit` package

## 7. Integration Testing (for significant changes)
- [ ] Test training pipeline if modifying trainer code
- [ ] Test inference if modifying model or inference code
- [ ] Verify checkpoint saving/loading if modifying checkpoint code

## Special Notes
- **No automatic formatting/linting**: The project does not appear to have black/flake8/mypy configured
- **No CI/CD detected**: Manual testing is important
- **Distributed training**: Test with appropriate GPU setup if modifying distributed components
- **Ray initialization**: Be aware that Ray is used for distributed execution
- **Hydra configs**: Changes to config structure may require updates in multiple places

## Common Pitfalls to Avoid
- Don't forget to set environment variables before running distributed training
- Ensure CUDA_VISIBLE_DEVICES is set appropriately for multi-GPU training
- Check that file paths in configs exist before running training
- Verify video file formats are compatible (decord, ffmpeg requirements)
- Be mindful of memory usage with large video files and models