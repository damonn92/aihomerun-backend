"""
独立测试脚本：测试视频处理 + 姿态分析 + 指标计算（不需要 API Key）
用法: python3 tests/test_video.py <视频路径> [swing|pitch] [年龄]
"""
import sys
import time
from pathlib import Path

# 加入项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import json


def test_video(video_path: str, action_type: str = "swing", age: int = 10):
    print(f"\n{'='*50}")
    print(f"🎬 视频: {video_path}")
    print(f"⚾ 动作: {action_type} | 年龄: {age}岁")
    print('='*50)

    # ── 1. 视频信息 ──
    from services.video_processor import extract_frames, get_video_info
    t0 = time.time()
    vpath = Path(video_path)
    info = get_video_info(vpath)
    print(f"\n📹 视频信息")
    print(f"   分辨率: {info['width']}x{info['height']}")
    print(f"   帧率:   {info['fps']:.1f} fps")
    print(f"   时长:   {info['duration_sec']:.1f} 秒")
    print(f"   总帧数: {info['total_frames']}")

    # ── 2. 抽帧 ──
    frames = extract_frames(vpath)
    print(f"\n🖼  抽帧完成: {len(frames)} 帧（目标 10fps）")

    # ── 3. MediaPipe 姿态分析 ──
    print("\n🦾 MediaPipe 姿态分析中...")
    from services.pose_analyzer import PoseAnalyzer
    analyzer = PoseAnalyzer()
    frames_data = analyzer.analyze_frames(frames)
    analyzer.close()

    valid = [f for f in frames_data if f is not None]
    print(f"   检测成功: {len(valid)}/{len(frames)} 帧 ({len(valid)/len(frames)*100:.0f}%)")

    if len(valid) < 3:
        print("❌ 有效帧太少，请确保：")
        print("   - 全身在画面内")
        print("   - 光线充足")
        print("   - 人物不要太小")
        return

    # 打印第一个有效帧的部分关键点（调试用）
    sample = valid[len(valid)//2]  # 取中间帧
    key_points = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
                  "LEFT_WRIST", "RIGHT_WRIST"]
    print("\n   关键点坐标样本（中间帧）:")
    for kp in key_points:
        if kp in sample:
            x, y, vis = sample[kp]
            print(f"   {kp:20s}: ({x:6.1f}, {y:6.1f})  可见度={vis:.2f}")

    # ── 4. 棒球指标计算 ──
    print(f"\n⚾ 计算棒球指标（{action_type}）...")
    from services.baseball_metrics import analyze_swing, analyze_pitch
    try:
        if action_type == "swing":
            metrics = analyze_swing(frames_data)
        else:
            metrics = analyze_pitch(frames_data)

        print(f"\n📊 运动指标")
        print(f"   动作类型:       {metrics.action_type}")
        print(f"   有效分析帧数:   {metrics.frames_analyzed}")
        print(f"   手腕峰值速度:   {metrics.peak_wrist_speed:.1f} px/帧")
        print(f"   髋肩分离角:     {metrics.hip_shoulder_separation:.1f}°")
        print(f"   重心平衡分:     {metrics.balance_score:.3f} (0-1)")
        print(f"   随挥完成:       {'✅ 是' if metrics.follow_through else '❌ 否'}")
        print(f"\n   关节角度:")
        print(f"   - 肘部角度:     {metrics.joint_angles.elbow_angle:.1f}°")
        print(f"   - 肩膀倾斜:     {metrics.joint_angles.shoulder_angle:.1f}°")
        print(f"   - 髋部旋转:     {metrics.joint_angles.hip_rotation:.1f}°")
        print(f"   - 膝盖弯曲:     {metrics.joint_angles.knee_bend:.1f}°")

    except ValueError as e:
        print(f"❌ 指标计算失败: {e}")
        return

    elapsed = time.time() - t0
    print(f"\n⏱  总耗时: {elapsed:.1f}s（不含 Claude API）")
    print("\n✅ 视频处理 & 姿态分析 正常！")
    print("   下一步：配置 ANTHROPIC_API_KEY 后可启动完整服务")

    return metrics


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    action = sys.argv[2] if len(sys.argv) > 2 else "swing"
    age = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    if not video:
        print("用法: python3 tests/test_video.py <视频路径> [swing|pitch] [年龄]")
        sys.exit(1)

    test_video(video, action, age)
