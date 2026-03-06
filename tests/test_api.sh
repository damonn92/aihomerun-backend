#!/bin/bash
# 快速测试脚本（需要先启动服务器）
# 用法: bash tests/test_api.sh <视频文件路径>

VIDEO=${1:-"test_swing.mp4"}
BASE_URL="http://localhost:8000"

echo "=== 健康检查 ==="
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== 上传分析（挥棒）==="
curl -s -X POST "$BASE_URL/analyze" \
  -F "file=@$VIDEO" \
  -F "action_type=swing" \
  -F "age=10" \
  | python3 -m json.tool
