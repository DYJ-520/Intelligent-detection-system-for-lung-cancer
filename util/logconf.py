import logging.handlers

root_logger = logging.getLogger()
# 设置根日志记录的级别
# 日志记录器的层级：DEBUG < INFO < WARNING < ERROR < CRITICAL
root_logger.setLevel(logging.INFO)

# 删除其他库的日志跟记录器
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
# 配置日志信息的格式
logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
# 设置控制台日志记录器的记录级别
streamHandler.setLevel(logging.INFO)
# 把控制台日志记录器添加到根日日志记录器中
root_logger.addHandler(streamHandler)