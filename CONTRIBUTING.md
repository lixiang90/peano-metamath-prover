# Contributing

1. 从新分支开始修改，不提交 `outputs/`、数据分片或模型检查点。
2. 所有新推理动作必须通过精确类型和 `$d` 约束检查。
3. 修改解析、组合、证书或验证逻辑时必须增加回归测试。
4. 声称“证明成功”时必须附带独立验证器重放结果。
5. 开放数学猜想不得作为成功监督标签。

提交前运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m metamath_generator formal/peano.mm \
  --steps 100 --output-dir outputs/smoke
```
