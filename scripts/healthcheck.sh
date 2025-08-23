#!/bin/bash
"""
HybrIK Server Health Check Script
Tests server health and basic functionality
"""

set -e

HOST=${1:-localhost}
PORT=${2:-8081}
BASE_URL="http://${HOST}:${PORT}"

echo "🏥 Health checking HybrIK Server at $BASE_URL..."

# Test 1: Basic health check
echo "📋 Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$BASE_URL/health" || echo "000")

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed"
    cat /tmp/health_response.json | python -m json.tool
else
    echo "❌ Health check failed (HTTP $HEALTH_RESPONSE)"
    cat /tmp/health_response.json 2>/dev/null || echo "No response body"
    exit 1
fi

echo ""

# Test 2: Image analysis with dummy image (if server is ready)
READY=$(cat /tmp/health_response.json | python -c "import sys, json; print(json.load(sys.stdin).get('ready', False))" 2>/dev/null || echo "False")

if [ "$READY" = "True" ]; then
    echo "🖼️  Testing /analyze-image endpoint with dummy image..."
    
    # Create a simple test image (1x1 white pixel base64)
    TEST_IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    ANALYZE_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/analyze_response.json \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"image_b64\":\"$TEST_IMAGE_B64\"}" \
        "$BASE_URL/analyze-image" || echo "000")
    
    if [ "$ANALYZE_RESPONSE" = "200" ]; then
        echo "✅ Image analysis test passed"
        echo "Response preview:"
        cat /tmp/analyze_response.json | python -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  - joints3d length: {len(data.get('joints3d', []))}\"
print(f\"  - processing time: {data.get('meta', {}).get('latency_ms', 'N/A')}ms\")"
    else
        echo "⚠️  Image analysis test failed (HTTP $ANALYZE_RESPONSE)"
        cat /tmp/analyze_response.json 2>/dev/null || echo "No response body"
        echo "This might be expected if HybrIK is not fully set up yet."
    fi
else
    echo "⚠️  Server not ready for image analysis (HybrIK not initialized)"
fi

echo ""
echo "🎉 Health check completed!"

# Cleanup
rm -f /tmp/health_response.json /tmp/analyze_response.json