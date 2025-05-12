set -xe

ckpt=$1
shift

if [ ! -f "$ckpt" ]; then
    echo "Error: ckpt file does not exist: $ckpt"
    exit 1
fi

# export DISPLAY=:1 # set DISPLAY variable if you run this in a machine with GUI

python3 ./motionblender/app/main.py \
    --ckpt-path $ckpt \
    --lazy-render-interval 0.5 --work-dir-from-ckpt True $@
