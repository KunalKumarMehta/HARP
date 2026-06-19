ssh -i <PRIVATE_KEY_FILE_PATH> -L 2222:sa653615.sa.svc.cluster.local:22 -N sshtunnel@ssh.qdc.qualcomm.com

ssh -i <PRIVATE_KEY_FILE_PATH> -o StrictHostKeychecking=no -o UserKnownHostsFile=/dev/null -L 5556:localhost:3389 -o StrictHostKeyChecking=no -p 2224 hcktest@localhost

reg add HKLM\SYSTEM\CurrentControlSet\Control\Lsa /v LimitBlankPasswordUse /t REG_DWORD /d 0 /f

localhost:5555

Snapdragon\HCKTest