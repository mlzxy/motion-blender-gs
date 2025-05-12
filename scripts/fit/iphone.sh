set -xe

scene=$1
shift
otherargs=$@

data_args="data:iphone --data.data-dir ./datasets/iphone/$scene 
    --data.cache-version jan18  --data.load-from-cache" # the cache-version can be arbitrary string

python ./motionblender/train.py  --work-dir outputs/iphone/$scene  --train-steps 30000  \
     --loss.w-smooth-motion 1.0 --motion-init-batch-size -1 --motion-init-steps 400 --skip-save-pv  $otherargs $data_args

# optionally, you can also run extra 10000 steps to refine the results, it may (or have no effect) help on certain scenes
python ./motionblender/train.py  --work-dir outputs/iphone/$scene  --train-steps 10000  \
    --loss.w-smooth-motion 1.0 --resume-if-possible  motion_pretrained  --skip-save-pv  \
    --lr.means 8e-5  --lr.opacities 2e-2 --lr.scales 5e-4  --lr.quats 5e-4 --lr.colors 5e-3  --lr.motions 1e-4 \
    --step-offset 30000  $otherargs $data_args
