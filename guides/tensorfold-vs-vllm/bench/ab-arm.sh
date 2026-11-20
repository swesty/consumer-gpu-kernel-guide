# usage: bash -s <0|1>  — set VLLM_ENABLE_ROCE_ALLREDUCE in the GLM recipe and reload via glm53-load.sh
V="$1"; R=$HOME/spark-vllm-docker/recipes/glm-5.3-flash-nvidia-nvfp4.yaml
python3 - "$V" "$R" <<'PY'
import re, sys
v, path = sys.argv[1], sys.argv[2]
t = open(path).read()
new, n = re.subn(r'(  VLLM_ENABLE_ROCE_ALLREDUCE: )"[01]"', r'\1"%s"' % v, t)
assert n == 1, n
open(path, "w").write(new)
PY
grep -n 'VLLM_ENABLE_ROCE_ALLREDUCE' "$R"
bash $HOME/glm53-load.sh > /dev/null 2>&1 < /dev/null; rc=$?
echo "glm53-load exit $rc"; tail -4 $HOME/glm53-load.log
docker logs vllm_glm53 2>&1 | grep -a -E "backends .* group 'tp:0'" | head -1 | cut -c1-160
awk '/MemAvailable/{print "spark-3 MemAvailable MiB", int($2/1024)}' /proc/meminfo
