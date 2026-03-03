# GEM Fan Club RAG — 开发记录与未来优化方向

---

## 未完成的优化项（按工作量从小到大排序）

### ⭐ 极小工作量（< 10 分钟）

> 以下已全部完成 ✅

---

### ⭐⭐ 小工作量（10-30 分钟）

#### 1. 歌词分块策略优化
- [ ] `_lyrics_splitter` 每首歌产生 2 个 chunk（song_info 摘要 + 完整歌词），存在冗余
- [ ] 歌曲信息被重复索引，检索时可能同时命中两个 chunk 占用名额
- [ ] 建议合并为一个结构化 chunk，或为 song_info 添加 `is_summary=True` 标记做后处理去重

#### 2. 歌词解析逻辑加固
- [ ] `_lyrics_splitter` 中 `line[0] == ' '` 只检查空格不含 tab，且未先 strip
- [ ] `line.replace('歌词', '')` 会错误替换歌词内容中包含"歌词"的行
- [ ] 歌曲名提取依赖文件格式严格一致，应增加容错

#### 3. 生涯分块——时间标题合并
- [ ] `_career_splitter` 将时间标题（如"2014年－2017年：脱离蜂鸟音乐后发展"）单独成块
- [ ] 短文本 chunk 语义信息不足，向量检索匹配质量差
- [ ] 建议将时间标题与其下方内容段落合并为一个 chunk

#### 4. RAG 服务端限流
- [ ] 当前 RAG FastAPI 端无任何请求频率限制
- [ ] Java 后端已有 Redis 限流，但如果 RAG 服务直接暴露则无防护
- [ ] 建议集成 `slowapi` 或 FastAPI 中间件，按 IP 限制每分钟请求数

#### 5. 多演唱会数据兼容性
- [ ] concert 目录当前仅一个 JSON 文件，`rglob("*.json")` 暗示支持多个
- [ ] `_enhance_metadata` 中根据文件名提取 `concert_name` 的逻辑依赖命名规范
- [ ] 建议在 JSON 数据内部新增 `concert_name` 字段，不依赖文件名

#### 6. 前端生产环境部署适配
- [ ] `api.js` 中 `baseURL` 硬编码为 `localhost:7071`
- [ ] `vite.config.js` 未配置 `server.proxy`
- [ ] 生产部署需要环境变量驱动或 Nginx 反代配置

---

### ⭐⭐⭐ 中等工作量（30-120 分钟）

#### 7. 问题重写与检索并行化
- [ ] 当前 `enhanced_rag_chain` 串行执行：问题重写 → 检索 → 生成
- [ ] BM25 检索可用原始问题先行启动，只有向量检索需要等重写后的问题
- [ ] 使用 `asyncio.gather` 或 `concurrent.futures` 实现部分并行，减少 1 次 LLM 延迟

#### 8. 知识库数据扩充
- [ ] 增加更多演唱会巡演数据（目前仅 I AM Gloria 一个 JSON）
- [ ] 补充专辑信息（发行日期、曲目列表、制作人等结构化数据）
- [ ] 考虑接入网易云/QQ 音乐公开 API 获取歌曲评论热度等辅助信息

---

### ⭐⭐⭐⭐ 大工作量（> 2 小时）

#### 9. 流式响应（SSE/Streaming）
- [ ] 新增 `/chat_gem/stream` 端点，使用 FastAPI `StreamingResponse` + SSE
- [ ] 智谱 AI 的 ChatZhipuAI 支持 `streaming=True`，逐 token 返回
- [ ] 前端 AIPage.vue 改用 EventSource 或 fetch stream 逐字渲染回答
- [ ] 用户体验从"等 5-15 秒才看到完整回答"提升为"即时打字机效果"

#### 10. 多媒体回复能力
- [ ] Bot 支持返回音频片段或图片
- [ ] 歌词 chunk 关联音频文件路径（简单方案：base64 编码截取一小段音频预览）
- [ ] 图片从微博爬取做转发展示，需封装为独立 skill

#### 11. 微博爬虫 Skill
- [ ] 添加微博爬虫模块，定时抓取邓紫棋相关最新简讯
- [ ] 将爬取内容结构化后注入知识库，保持 RAG 数据时效性
- [ ] 前端展示"最新动态"板块

---

## 已完成的改进 ✅

- [x] **v2.0 Agent 改造（Phase 1）** — 从硬编码路由升级为 LLM 自主决策的 Tool-Augmented Agent
  - [x] `agent/gem_agent.py`：Agent 编排核心（问题重写→工具规划→工具执行→回答生成）
  - [x] `agent/tools/tool_registry.py`：工具注册中心
  - [x] `agent/tools/knowledge_tools.py`：3 个知识检索工具
  - [x] `agent/tools/time_tools.py`：时间工具
  - [x] `agent/tools/recommend_tools.py`：歌曲推荐工具
  - [x] `rag_server.py` 双模式运行（`--mode agent|rag`）
  - [x] `config.py` Agent 配置项
  - [x] `CHANGELOG.md` 版本迭代文档
- [x] `_clean_text` 过度清洗 → 改为只清除不可见控制字符，保留所有正常标点
- [x] `hot_song.json` 集成到知识库（50 首热门歌曲可被 RAG 检索）
- [x] `rag_modules/__init__.py` 创建（标准 Python 包）
- [x] Prompt 模板无效空行清理（节省约 20-30 Token/次）
- [x] `_enhance_metadata` 不覆盖已有 category
- [x] RestTemplate 超时配置（5s 连接 + 90s 读取）
- [x] AI 接口 IP 限流保护（每 IP 每 60s 最多 5 次）
- [x] 前端超时调整为 120s（对齐 RAG 服务超时链路）
- [x] 前端 history 遍历 bug 修复（逐条检查防止 undefined）
- [x] `manageContext` / `shouldTerminateConversation` 矛盾修复
- [x] `_enhance_metadata` 死代码清理（移除未使用的 `path_parts`、无效的 `endswith` 检查）
- [x] `data_preparation.py` 所有 `print` 统一改为 `logger`
- [x] 时间关键词检测扩展（新增英文词 + 年份匹配 `20xx`）
- [x] 文档去重策略改用 `chunk_id` 作为唯一标识（替代 `page_content[:200]`）
- [x] 检索 K 值调优（30→10，HYBRID_RETRIEVER_NUM_RESULTS 10→5）
- [x] 对话历史截断轮数提取为配置项 `HISTORY_MAX_TURNS`
- [x] 后端 JPA `ddl-auto` 配置矛盾修复（移除冲突的 `hbm2ddl.auto: update`）