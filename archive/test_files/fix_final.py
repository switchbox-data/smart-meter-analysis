with open("smart_meter_analysis/census.py") as f:
    lines = f.readlines()

# Find and fix the _import_cenpy line
for i, line in enumerate(lines):
    if "def _import_cenpy()" in line:
        lines[i] = "def _import_cenpy():  # type: ignore[no-untyped-def]\n"
        break

with open("smart_meter_analysis/census.py", "w") as f:
    f.writelines(lines)
