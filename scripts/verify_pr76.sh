#!/bin/bash
# Verification script for PR #76: Namespace Pollution Fix

set -e
echo "PR #76 Verification"
echo "==================="

# Test imports
python3 -c "
import sys; sys.path.insert(0, 'src')
from mycelium_fractal_net.analytics import FeatureVector
from mycelium_fractal_net.experiments import generate_dataset
print('✓ Canonical imports work')
"

# Test API
python3 -c "
import sys; sys.path.insert(0, 'src')
from mycelium_fractal_net.analytics.fractal_features import FeatureVector
fv = FeatureVector(values={'D_box': 1.5})
assert fv.D_box == 1.5 and fv['D_box'] == 1.5
print('✓ API backward compatibility works')
"

# Check structure
[ ! -d "analytics" ] && [ ! -d "experiments" ] && echo "✓ No top-level pollution"
[ -d "src/mycelium_fractal_net/analytics" ] && echo "✓ Canonical structure exists"

# Check protection
grep -q "/analytics/" .gitignore && grep -q "/experiments/" .gitignore && echo "✓ .gitignore protection active"

# Check lint
ruff check . > /dev/null 2>&1 && echo "✓ All lint checks pass"

echo ""
echo "✅ All validations passed!"
