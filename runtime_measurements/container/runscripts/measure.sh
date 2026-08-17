#!/bin/bash

export TIMEFORMAT=$'########\nREALTIME=%3R\nUSERTIME=%3U\nSYSTIME=%3S\n'

sleep 2

start=$(date '+%s.%N')

time source start.sh "$@"

end=$(date '+%s.%N')

echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "TS_RUNTIME=${runtime}\n########"
