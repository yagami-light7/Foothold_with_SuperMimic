"""Isaac Lab/Newton仿真检查，按README中的命令单独运行。"""


def load_tests(loader, standard_tests, pattern):
    """阻止unittest递归导入CLI仿真脚本；这些文件在导入时会解析启动参数。"""
    return standard_tests
