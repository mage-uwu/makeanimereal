#!/bin/bash
# Talk to the pod's server (coverage/pod/server.py) through the RunPod proxy.
#   POD=<id> POD_TOKEN=<token> coverage/pod/pod.sh put <local> <remote> | get <remote> <local> | run "<cmd>" | cat <remote> | ls [dir]
U="https://$POD-8000.proxy.runpod.net"; H="X-Token: $POD_TOKEN"
case "$1" in
  put) curl -sS -f -H "$H" -T "$2" "$U/f/$3" ;;
  get) curl -sS -f -H "$H" -o "$3" "$U/f/$2" ;;
  cat) curl -sS -H "$H" "$U/f/$2" ;;
  run) curl -sS -f -H "$H" --data-binary "$2" "$U/run" ;;
  ls)  curl -sS -f -H "$H" "$U/ls/$2" ;;
esac
