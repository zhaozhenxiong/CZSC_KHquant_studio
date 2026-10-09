# 当前目录

- vendor/czsc：附件冻结源码与许可证
- vendor/wheels：校验过的本机原生核心
- my_strategy/adapters：原始日线及来源边界
- my_strategy/services：结构、事件、仓位与回测
- my_strategy/execution：BrokerSimulator 与 Ledger
- my_strategy/storage：原始行情、CZSC 结果索引及持久关注/手动持仓
- my_strategy/web_dashboard：六入口网页、API、任务
- my_strategy/core：配置、路径、运行上下文
- my_strategy/configs：当前 CZSC 与行情配置
- my_strategy/scripts：更新、安装验证、打包、审计
- my_strategy/knowledge_base：追加的历史审计与当前 ADR

运行数据不纳入源码包，业务路径来自仓库 .env。
