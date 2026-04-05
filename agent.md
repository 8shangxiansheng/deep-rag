# Agent Push Log

## 2026-04-05 15:55:32 +0800
- 任务：P0-1 配置接口安全收敛（默认关闭、读脱敏、写入需本机来源 + 管理员令牌）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：5c1d8ad6a3016d3cb071ce71a5b2a7cb935c64d5
- 推送：`git push origin main`

## 2026-04-05 15:56:48 +0800
- 任务：P0-2 收紧 CORS 白名单（改为配置化来源列表）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：56a464505aa08e92f0bfd3e1f8ab4734b95cd7d4
- 推送：`git push origin main`

## 2026-04-05 15:59:02 +0800
- 任务：P0-3 增加检索安全边界（路径防逃逸、全量检索开关、文件与字符上限）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：c1fb8423d5af028ee169da77904daca764828239
- 推送：`git push origin main`

## 2026-04-05 16:00:46 +0800
- 任务：P0-4 关闭敏感调试输出（改为可控日志）并收敛对外错误信息
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：de668d53044a49552ee1d4d69c466b93e8088ccb
- 推送：`git push origin main`

## 2026-04-05 16:06:17 +0800
- 任务：P1-1 统一前端配置接口基址（移除硬编码 localhost）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：f469e4541893a2f11755ffb09b8321a108d70c5a
- 推送：`git push origin main`

## 2026-04-05 16:09:05 +0800
- 任务：P1-2 增加 `/chat` 与 `/knowledge-base/retrieve` 的限流，并为上游 LLM 调用增加超时/重试/熔断，统一错误码结构
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：f128b97dd20db4ad78e58ab2509a9a83fafeef51
- 推送：`git push origin main`

## 2026-04-05 16:10:23 +0800
- 任务：P1-3 补充最小自动化测试集（限流器、检索边界、错误结构契约）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：4175a38e5c76d4089961ec2366cf6e37b4f3c14b
- 推送：`git push origin main`

## 2026-04-05 16:15:27 +0800
- 任务：P2-1 重构 ReAct 解析为结构化协议，并兼容旧版协议
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：d8868346f4888fa32478186c3c99269fa0123b92
- 推送：`git push origin main`

## 2026-04-05 16:20:42 +0800
- 任务：P2-2 建立 CI 质量门禁（lint/test/security scan）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：4deb834e3425c383d913d3f75d099079945a7ada
- 推送：`git push origin main`

## 2026-04-05 16:25:30 +0800
- 任务：P2-3 摘要生成流程产品化（可复现流水线、失败重试、产物版本化）
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：02e7c2003f5432def8d594d896c6861060cb39fe
- 推送：`git push origin main`

## 2026-04-05 16:53:58 +0800
- 任务：Review 修复 1：修复 ReAct 旧协议解析贪婪匹配导致的工具调用漏触发
- 分支：main
- 远程：https://github.com/8shangxiansheng/deep-rag.git
- 提交：2175e89bae7213e1e9afdb556bbe8a40b97e3562
- 推送：`git push origin main`
