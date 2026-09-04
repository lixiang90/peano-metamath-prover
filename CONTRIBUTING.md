# Contributing

1. 从新分支开始修改，不提交 `outputs/`、数据分片或模型检查点。
2. 所有新推理动作必须通过精确类型和 `$d` 约束检查。
3. 修改解析、组合、证书或验证逻辑时必须增加回归测试。
4. 声称“证明成功”时必须附带独立验证器重放结果。
5. 开放数学猜想不得作为成功监督标签。

提交前运行：

```bash
python -m pip install -e ".[neural,dev]"
python -m pytest
python -m metamath_generator formal/peano.mm \
  --steps 100 --output-dir outputs/smoke
```

命令在仓库根目录执行；以上为 Bash 续行语法。已安装 PyTorch 的本地 PowerShell
环境也可运行 `$env:PYTHONPATH='src'` 后执行 `python -m unittest discover -s tests`。
仅安装 `.[dev]` 时，缺少 PyTorch 会使部分神经测试跳过，不能作为完整回归通过。

修改 PA+ 定义时还须运行 `tests/test_pa_plus.py`，并核对生成的 `.mm` 与 JSON 目录
一致。更新文档时应区分历史实验、当前实现和未来计划，核对 CLI 参数及证书验证
口径；不要把目标触达数、token accuracy 或工程 smoke 记成目标证明成功率。
