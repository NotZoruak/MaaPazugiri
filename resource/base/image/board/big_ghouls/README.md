# 大讨伐战斗识别配置

本配置适配用户提供的“大討伐”活动战斗画面，支持同一棋盘上大小不同的普通四色饿鬼。只处理战斗内识别，不负责活动识别、菜单导航或任务界面，普通战斗仍使用 `development/profile.json`。

## 使用

在项目根目录运行：

```powershell
uv run python tools/board_vision.py analyze --profile resource/base/image/board/big_ghouls/profile.json --input training/board-samples/events/big-ghouls/raw/001_mixed_sizes.png training/board-samples/events/big-ghouls/raw/002_mixed_sizes_flame_item.png --output debug/board-vision/big-ghouls/final
```

现有 `BoardRecognizer` 和 `BoardPreviewer` 均接收配置路径；后续活动任务可显式传入本目录的 `profile.json`。本次未修改战斗动作默认路径或增加任务开关。

## 范围与方法

- 参考截图为720×1280，搜索区域 `[0, 280, 720, 710]` 依据两幅实机图核对，排除顶部蓝色敌人、紫色特效和下方角色。该区域不是对象坐标或固定网格，主体框可伸出搜索区域。
- 复用既有眼睛、嘴巴局部模板及来源记录。普通饿鬼独立搜索0.8至2.2倍、步长0.1的15档尺度，每只保留自身尺寸。此范围是搜索参数，不代表任意大小都已验证。
- 只对本配置启用跨尺度部位合并。相同模板在同一可见部位上的重复命中归到一个对象；相邻对象仍需满足中心和部位的几何关系。中心估计仅采用与最强证据主体尺寸相差不超过20%的命中，较小嘴巴命中保留为证据。
- 炸弹保留既有外焰模板、角色忽略掩码及0.9、1.0、1.1三档尺度，不随普通饿鬼一起扩大。第二幅图按现有外焰模板输出二级炸弹，开发参考只确认一个炸弹对象，等级尚未由该活动独立标注。
- 按用户确认的活动范围，跳过所有障碍物、特殊黑色饿鬼及强化外观；结果分别写入 `coverage.skipped_kinds`、`skipped_labels` 与 `skipped_variants`。这些类别未检查，不能将跳过理解为通用场景下已证明不存在。
- 小尺寸未匹配色块的提示门槛随最小搜索尺度调整；未知区域仍保留未知，不当作空地。
- 每帧独立定位，不保存或复用上一帧对象坐标。普通配置的窄尺度与原合并规则保持不变。

## 当前验证

两幅不同画面均为开发资料，前三份重复附件只保留一份。原图复制后逐字节哈希核对，未改动原始截图。开发参考位于 `training/board-vision/events/big-ghouls/samples.json`，近似中心用于逐个关联，不能充当精确主体分割标注或独立评估。

| 开发画面 | 红  | 黄  | 蓝  | 绿  | 炸弹 |
| -------- | --- | --- | --- | --- | ---- |
| 001      | 8   | 6   | 12  | 3   | 0    |
| 002      | 7   | 2   | 14  | 6   | 1    |

单个可见部位支撑的对象仍标为 `single_feature`；身体大小来自局部模板比例，不是精确轮廓。两张静态图不能验证动画期间、其他战斗场次或超出尺度范围的外观。结果继续保持 `automatic_action_allowed=false`，本次没有进行实机触控测试。
