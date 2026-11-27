# 历史基准评测

[index.json](index.json) 列出五份可归档的历史运行结果及其评测步数。CSV 包含原归档中的 `test-legacy` 和 `test-train-sampling` 核心标量。以运行、全局步、评测口径和指标为键，同键采用事件序号最新的记录。不同评测口径应分别比较。

其中四份记录含 step 217 的评测，包括 DeepSeek 的 vanilla OPD 和 ATI-OPD、Nemotron 的 vanilla OPD 和 ATI-OPD。另有一份 DeepSeek ATI-OPD 记录截至 step 120。此次同步的归档没有 FAD 系列的完整训练结果。
