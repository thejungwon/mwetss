# copy this to root directory of data and ./normalize-resample.sh
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}


path=$1
N=$2 # set "N" as your CPU core number.
open_sem $N
for ((i=1;i<=$N;i++)); do
    run_with_lock python generator.py $path $N $i
done
