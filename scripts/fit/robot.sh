set -xe

scene=$1
shift
extra_opts=$@

workdir=outputs/robot/$scene
train_opts="--train-steps 40000"

case $scene in
    robot)
        canoid=323
        ;;
    microwave)
        canoid=0
        humanid=2
        cameras="center left right"
        ;;
    rope)
        canoid=0
        humanid=1
        cameras="center"
        ;;  
    cloth)
        canoid=0
        humanid=1
        cameras="center left right"
        ;;  
    *)
        echo "Invalid scene: $scene"
        exit 1
        ;;
esac

data_opts="data:general  --data.data-dir  ./datasets/robot/$scene"
if [ -n "$humanid" ]; then
    data_opts=$data_opts" --data.mask-insts $humanid"
fi
data_opts=$data_opts" --data.depth-type metric_depth --data.no-use-tracks --data.no-normalize-scene  --data.use-median-filter --data.given-cano-t $canoid"


camera_opts=""
if [ -n "$cameras" ]; then
    camera_opts="--cameras $cameras"
fi


basic_opts="--fg-only  --loss.w-smooth-motion 1.0  $camera_opts 
    --ctrl.stop-control-by-screen-steps 4000 --ctrl.stop-control-steps 4000 
    --lr.annealing-for means scales motions   
    --lr.annealing-factor 1e-2"

case $scene in
    robot)
        basic_opts=$basic_opts" --loss.w-smooth-motion 0.0 --loss.w-sparse-link-assignment 0.0  --loss.w-minimal-movement-in-cano 0.0  --loss.w-rgb 2.0"
    ;;
    microwave)
        basic_opts=$basic_opts" --loss.w-minimal-movement-in-cano 0.0  --loss.w-rgb 2.0"
    ;;
    rope)
        basic_opts=$basic_opts" --loss.w-minimal-movement-in-cano -1.0  --loss.w-rgb 4.0  --loss.w-kp2d 1.0 --motion-init-batch-size -1  --motion-pretrain-with-kp3d  --no-motion-pretrain-with-means --motion-init-steps 400  --init-gamma 50"
    ;;
    cloth)
        basic_opts=$basic_opts" --loss.w-minimal-movement-in-cano -1.0  --loss.w-rgb 4.0   --loss.w-length-reg 0.5  --loss.length-reg-names cloth  --loss.w-arap 0.3"
    ;;
esac


python ./motionblender/train.py  --work-dir $workdir  $basic_opts $train_opts $extra_opts  $data_opts
