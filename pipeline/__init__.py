"""会话化流水线包：模块化步骤 + 注册表 + 执行上下文。

import 顺序留意：steps_* 在 import 时即 `import config`，而 config 在 import
时读取 os.environ。因此 run_session.py 必须在 import 本包**之前**写好 os.environ。
"""
