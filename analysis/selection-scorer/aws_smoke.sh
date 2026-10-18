#!/bin/bash
# Live AWS smoke of the selection scorer (mockllm only, no model spend): plans/v7-selscorer-smoke.tsv through the real
# scripts/v6.sh (agent scorer c7a.8xlarge via SCORER_MACHINE, selection scorer c7a.8xlarge by default). Every 60 s a
# sampler records both scorers' security-group ingress rules and every tool-tagged instance into smoke/aws-samples.txt,
# so the record shows when each scorer came up and went down and which IPs each scorer's port was open to.
# Run under bgjob: bgjob run --name selsmoke --grace 120 -- /usr/bin/caffeinate -is bash analysis/selection-scorer/aws_smoke.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=analysis/selection-scorer/smoke
mkdir -p "$OUT"
rm -f "$OUT/.done"
sample() {
  local n sg
  echo "== $(date -u +%FT%TZ)"
  for n in agent select; do
    (
      SCORER_NAME=$n
      source scripts/cloud/common.sh
      sg=$(aw ec2 describe-security-groups --filters "Name=group-name,Values=$AWS_SG_SCORER" --query 'SecurityGroups[0].GroupId')
      echo "$n $AWS_SG_SCORER ($sg): $(aw ec2 describe-security-group-rules --filters "Name=group-id,Values=$sg" \
        --query 'SecurityGroupRules[?!IsEgress].[CidrIpv4,Description]' 2>/dev/null | tr '\t\n' ' ;')"
    )
  done
  (
    source scripts/cloud/common.sh
    aw ec2 describe-instances --filters "Name=tag:tool,Values=$TOOL_LABEL" \
      Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`]|[0].Value,LaunchTime]'
  )
}
(
  for i in $(seq 1 180); do  # at most 3 h
    sample >> "$OUT/aws-samples.txt" 2>&1
    [ -e "$OUT/.done" ] && break
    sleep 60
  done
) &
sampler=$!
echo "[selsmoke] $(date +%H:%M:%S) start"
SEED_OFFSET_FILE=$HOME/projects/swarm-scaling/data/.algotune_seed_offset.v7 EXPERIMENT=v7 HOURS=4 SCORER_MACHINE=c7a.8xlarge \
  bash scripts/v6.sh plans/v7-selscorer-smoke.tsv 1000000000 selscorer --group-start --validity-n 20 \
  --script scripts/smoke_v7b.py --max-agents 5
rc=$?
echo "[selsmoke] $(date +%H:%M:%S) v6.sh exit $rc"
touch "$OUT/.done"
wait "$sampler"
sample >> "$OUT/aws-samples.txt" 2>&1
source scripts/cloud/common.sh
{
  echo "== $(date -u +%FT%TZ) every us-east-1 instance not terminated, then every tool-tagged volume"
  aw ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]'
  aw ec2 describe-volumes --filters "Name=tag:tool,Values=$TOOL_LABEL" --query 'Volumes[].[VolumeId,State]'
  echo "== tool-tagged instances, terminated included, with launch time and the reason for their state"
  aw ec2 describe-instances --filters "Name=tag:tool,Values=$TOOL_LABEL" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`]|[0].Value,LaunchTime,State.Name,StateTransitionReason]'
} > "$OUT/final-aws.txt" 2>&1
cat "$OUT/final-aws.txt"
exit $rc
