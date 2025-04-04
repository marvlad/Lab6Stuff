#!/bin/bash

DIR="./Results"
CONFIG_FILE="configfiles/ReadOutChain/ToolChainConfig"
CURRENT_TIME=$(date +%s)

set_run() {
    source Setup.sh
    ./main configfiles/ReadOutChain/ToolChainConfig
}

update_inline() {
    #sed -i '27s/.*/Inline '"$1"'/' $CONFIG_FILE
    sed -i '' '27s/.*/Inline '"$1"'/' $CONFIG_FILE
}

show_oscilloscope(){
    latest_file=$(ls -lrt $DIR/*.txt |tail -n 1 |rev | cut -d'/' -f1 | rev)
    echo "Last file: $latest_file"
    path_last_file=$(ls -lrt $DIR/*.txt | tail -n 1 | awk '{print $9}')
    #filetime=$(stat --format="%Y" $path_last_file)
    filetime=$(stat -f "%m" $path_last_file)
    diff=$(echo $CURRENT_TIME - $filetime |bc)
    echo "current time: '$CURRENT_TIME'"
    echo "file time: '$filetime'"
    echo "diff time: '$diff'"
    if [ "$diff" -lt $1 ]; then
        echo "You have a new file!"
        SimpleScope.py $path_last_file --voltage --baseline --plot 10
        rm -f $path_last_file
    else
        echo "Sorry, I can't find a new file." 
    fi
}
# Parse arguments
for arg in "$@"; do
    case $arg in
        --test)
            update_inline 20
            # set_run
            show_oscilloscope 20
            ;;
        --data)
            update_inline 1000
            # set_run
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: source daq.sh [--test | --data]"
            return 1
            ;;
    esac
done
