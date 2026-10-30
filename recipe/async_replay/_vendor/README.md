# 项目内工具依赖

qiuputils 来自原工作区 packages/qiuputils/qiuputils，保留原实现。导入统一使用 recipe.async_replay._vendor.qiuputils，避免依赖项目外的 editable install，也不修改 sys.path 或注册全局 qiuputils 别名。

训练和 SFT 使用日志、数据路径展开、随机种子及 checkpoint 工具。PyYAML 等第三方依赖仍由 Python 环境提供。源包 README 和 pyproject 元数据一并保留以记录出处；它们不是新增的安装步骤。后续修改仅影响此副本，不会修改原工作区 packages 目录。
