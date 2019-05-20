#/bin/bash
while true;
do
    python ${1}_api.py &> log${1}API$(date +%Y-%m-%d).txt;
    echo "["$(date)"] ${1} API crashed." >> log${1}API_keep_alive.txt;
    sleep 5;
done

