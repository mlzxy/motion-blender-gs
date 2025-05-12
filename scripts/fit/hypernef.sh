set -xe
scene=$1
shift
extra_opts=$@

workdir=outputs/hypernerf/$scene

case $scene in
    vrig-peel-banana)
        skip_preprocess_pcd_clustering="person-1"
        ;;
    vrig-3dprinter)
        skip_preprocess_pcd_clustering="object-1"
        ;;
    vrig-chicken)
        skip_preprocess_pcd_clustering="person-2"
        ;;
    broom2)
        skip_preprocess_pcd_clustering="person-1"
        ;;
    *)
        echo "Invalid scene: $scene"
        exit 1
        ;;
esac

data_opts="data:general 
        --data.data-dir  ./datasets/hypernerf/${scene} 
        --data.load-from-cache --data.cache-version jan18   --data.img-path-suffix  2x 
        --data.depth-type aligned_depth_anything --data.use-tracks --data.normalize-scene  --data.K-scale 2"

train_opts="--train-steps 30000"

basic_opts="--loss.w-smooth-motion 1.0  --cameras left1 right1 --camera-name-override left1:,right1: --loss.w-rgb 1.0 
    --deformable-link-quantize 100 --num-vertices-for-deformable 50  
    --ctrl.stop-control-by-screen-steps 5000 --ctrl.stop-control-steps 5000  --camera-adjustment 
    --lr.annealing-for means scales motions  --init-with-rgbd" # --output-val-images

if [ "$scene" == "vrig-peel-banana" ]; then
    basic_opts="$basic_opts --loss.w-depth-grad 0 --loss.w-depth 0 --loss.w-depth-track 0"
fi

python motionblender/train.py  --work-dir $workdir  $basic_opts $train_opts  $extra_opts  $data_opts